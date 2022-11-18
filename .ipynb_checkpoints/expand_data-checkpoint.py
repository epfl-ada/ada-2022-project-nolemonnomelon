##################################################################
# This script brings together IMDB data with associated wikipedia data 
# and notably box office values from the CMU dataset to create a big 
# representative movie dataset, used for large-scale analysis.
##################################################################

import pandas as pd

# Load IMDB ratings
IMDB_data = pd.read_csv("IMDB_data/title.ratings.tsv", sep='\t')

# Load IMDB "US" versions of titles
IMDB_akas = (pd.read_csv("IMDB_data/title.akas.tsv", sep='\t',
                     usecols=['titleId', 'title', 'region'])[lambda x: x['region'] == 'US'])
# Load year of release
IMDB_basics = (pd.read_csv("IMDB_data/title.basics.tsv", sep='\t',
                      usecols=['tconst', 'startYear', 'genres', 'titleType'])[lambda x: x['titleType'] == 'movie'])

# Use key to merge IMDB movie titles and ratings
IMDB_data = IMDB_data.set_index('tconst').join(IMDB_akas.set_index('titleId')['title'], how='outer')

# Use key to merge title, rating, year of release of movies only
IMDB_data = IMDB_basics.set_index('tconst').drop(labels=['titleType'], axis = 1).join(IMDB_data, how='left')

# remove film duplicates due to various titles used for same movie
IMDB_data = IMDB_data[~IMDB_data.index.duplicated(keep='first')]


# transform the column genres from string to a list of strings
def list_genres(s) :
    if isinstance(s, str) :
        return list(s.split(','))
    return float('NaN')
IMDB_data['genres'] = IMDB_data['genres'].apply(lambda x : list_genres(x))

# MERGE wikidata with IMDB;
wiki = pd.read_csv('Expanded_data/wikipedia_query.tsv', sep='\t')
IMDB_wiki_data = IMDB_data.join(wiki.set_index('imdb_id.value')[['revenue.value', 'freebaseID.value']], how='left').reset_index()

# assert there is 1 row per imdb identifier for security
assert IMDB_wiki_data.index.duplicated().sum() == 0, "IMDB_wiki contains duplicates"


""""""""""""""""""""""""
"""   INFO exchange  """
""""""""""""""""""""""""

# Load CMU dataset
columns_movie = ['Wikipedia movie ID', 'Freebase movie ID', 'Movie name', 'Movie release date', 'Movie box office revenue',
                 'Movie runtime', 'Movie languages (Freebase ID:name tuples)', 'Movie countries (Freebase ID:name tuples)',
                 'Movie genres (Freebase ID:name tuples)']
df_movie = pd.read_csv("MovieSummaries/movie.metadata.tsv",sep='\t', names=columns_movie)

# rename columns for correspondance between dataframes
IMDB_wiki_data.rename(columns={'index' : 'IMDB_id', 'revenue.value': 'Movie box office revenue',
                               'freebaseID.value': 'Freebase movie ID', 'title' : 'Movie name',
                               'startYear' : 'Movie release date', 'genres' : 'Movie genres names'}, inplace=True)

# # transfer release dates and box office revenue to CMU dataset
# df_movie = df_movie.set_index('Freebase movie ID').combine_first(IMDB_wiki_data.set_index('Freebase movie ID')[['Movie release date', 'Movie box office revenue']]).reset_index()
# # remove non-originally-CMU rows
# df_movie = df_movie.loc[df_movie['Movie name'].notna()]
# # remove movie duplicated by the combination
# df_movie = df_movie[~df_movie['Freebase movie ID'].duplicated(keep='first')]

# transfer movie names, release dates and box office revenue from CMU dataset to IMDB_wiki
IMDB_wiki_data = IMDB_wiki_data.set_index('Freebase movie ID').combine_first(df_movie.set_index('Freebase movie ID')[['Movie name', 'Movie release date', 'Movie box office revenue']]).reset_index()

# reorder columns in dataframes
# df_movie = df_movie[columns_movie]

cols = ['IMDB_id', 'Freebase movie ID', 'Movie name',
       'Movie release date', 'Movie genres names',
       'averageRating', 'numVotes', 'Movie box office revenue']

IMDB_wiki_data = IMDB_wiki_data[cols]

# Save expanded IMDB_wiki
IMDB_wiki_data.to_csv('Expanded_data/big_data.tsv', sep='\t', index = False)

# Save expanded movie dataset
# df_movie.to_csv('Expanded_data/movie.expanded_metadata.tsv', sep='\t', index = False)