##################################################################
# This scripts brings together IMDB data with associated wikipedia data to create
# a representative movie dataset, used for large-scale analysis
# Then, it exchanges complementary information between the CMU data and the newly created data.
##################################################################

import pandas as pd

# Load IMDB ratings
IMDB_data = pd.read_csv("IMDB_data/title.ratings.tsv", sep='\t')

# Load IMDB "US" versions of titles
IMDB_akas = (pd.read_csv("IMDB_data/title.akas.tsv", sep='\t',
                     usecols=['titleId', 'title', 'region'])[lambda x: x['region'] == 'US'])
# Load year of release
IMDB_basics = pd.read_csv("IMDB_data/title.basics.tsv", sep='\t',
                      usecols=['tconst', 'startYear', 'genres'])

# Use key to merge IMDB movie titles and ratings
IMDB_data = IMDB_data.set_index('tconst').join(IMDB_akas.set_index('titleId')['title'], how='outer')

# Use key to merge title, rating, year of release
IMDB_data = IMDB_data.join(IMDB_basics.set_index('tconst'), how='outer')

# remove film duplicates due to various titles used for same movie
IMDB_data = IMDB_data[~IMDB_data.index.duplicated(keep='first')]

# divide genres into 3 priority columns
genres = IMDB_data.genres.str.split(',', expand=True)
IMDB_data.drop('genres', axis='columns', inplace=True)
IMDB_data['genre1'] = genres[0]
IMDB_data['genre2'] = genres[1]
IMDB_data['genre3'] = genres[2]


# MERGE wikidata with IMDB; new df contains :
wiki = pd.read_csv('Expanded_data/wikipedia_query.tsv', sep='\t')
# 'Freebase movie ID', 'Movie name', 'Movie release date', 'genre1',
# 'genre2', 'genre3', 'averageRating', 'numVotes', 'Movie box office revenue'
IMDB_wiki_data = IMDB_data.join(wiki.set_index('imdb_id.value')[['revenue.value', 'freebaseID.value']], how='left')

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
IMDB_wiki_data.rename(columns={'revenue.value': 'Movie box office revenue', 'freebaseID.value': 'Freebase movie ID',
                               'title' : 'Movie name', 'startYear' : 'Movie release date'}, inplace=True)

# transfer release dates and box office revenue to CMU dataset
df_movie = df_movie.set_index('Freebase movie ID').combine_first(IMDB_wiki_data.set_index('Freebase movie ID')[['Movie release date', 'Movie box office revenue']]).reset_index()
# remove non-originally-CMU rows
df_movie = df_movie.loc[df_movie['Movie name'].notna()]
# remove movie duplicated by the combination
df_movie = df_movie[~df_movie['Freebase movie ID'].duplicated(keep='first')]

# transfer movie names, release dates and box office revenue from CMU dataset to IMDB_wiki
IMDB_wiki_data = IMDB_wiki_data.set_index('Freebase movie ID').combine_first(df_movie.set_index('Freebase movie ID')[['Movie name', 'Movie release date', 'Movie box office revenue']]).reset_index()

# reorder columns in dataframes
df_movie = df_movie[columns_movie]

cols = ['Freebase movie ID', 'Movie name',
       'Movie release date', 'genre1', 'genre2', 'genre3',
       'averageRating', 'numVotes', 'Movie box office revenue']
IMDB_wiki_data = IMDB_wiki_data[cols]

# Save expanded IMDB_wiki
IMDB_wiki_data.to_csv('Expanded_data/IMDB_wiki.tsv', sep='\t', index = False)

# Save expanded movie dataset
df_movie.to_csv('Expanded_data/movie.expanded_metadata.tsv', sep='\t', index = False)