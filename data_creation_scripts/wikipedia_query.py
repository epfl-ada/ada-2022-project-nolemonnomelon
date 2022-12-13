from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd

# Request imdb-associated films on wikipedia, with associated box office revenues and freebase IDs if available (on wikipedia)
# freebase IDs will allow matching with our original dataset if needed later on

# Call the wikidata query service
sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

# create query
sparql.setQuery("""
SELECT ?film ?filmLabel ?imdb_id ?revenue ?freebaseID ?budget
WHERE
{
    ?film wdt:P31 wd:Q11424 .
    ?film wdt:P345 ?imdb_id .
    OPTIONAL {?film wdt:P2142 ?revenue .}
    OPTIONAL {?film wdt:P2130 ?budget .}
    OPTIONAL {?film wdt:P646 ?freebaseID .}

    SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
}
""")

# launch query and convert into pandas dataframe
# (This part can crash sometimes, probably because of wikipedia itself)
sparql.setReturnFormat(JSON)
results = sparql.query().convert()
wiki = pd.json_normalize(results['results']['bindings'])
wiki = wiki[['filmLabel.value', 'freebaseID.value', 'imdb_id.value', 'revenue.value', 'budget.value']]

# queries have duplicates because box office revenue is sometimes
# calculated for different countries
# we only keep the first one, which is worldwide
wiki = wiki[~wiki['imdb_id.value'].duplicated(keep='first')]

# Save wikipedia query
wiki.to_csv('data/Expanded_data/wikipedia_query.tsv', sep='\t', index=False)
