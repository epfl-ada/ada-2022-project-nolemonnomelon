import pandas as pd

# Load our movie dataset
columns_movie = ['Wikipedia movie ID', 'Freebase movie ID', 'Movie name', 'Movie release date', 'Movie box office revenue',
                 'Movie runtime', 'Movie languages (Freebase ID:name tuples)', 'Movie countries (Freebase ID:name tuples)',
                 'Movie genres (Freebase ID:name tuples)']
df_movie = pd.read_csv("MovieSummaries/movie.metadata.tsv",sep='\t', names=columns_movie)
# Load IMDB ratings
IMDB_ratings = pd.read_csv("IMDB_data/title.ratings.tsv", sep='\t')
# Load IMDB "US" versions of titles
IMDB_akas = (pd.read_csv("IMDB_data/title.akas.tsv", sep='\t',
                     usecols=['titleId', 'title', 'region'])[lambda x: x['region'] == 'US'])
# Load year of release
IMDB_basics = pd.read_csv("IMDB_data/title.basics.tsv", sep='\t',
                      usecols=['tconst', 'startYear'])

# Use key to merge IMDB movie titles and ratings
IMDB_ratings = IMDB_ratings.set_index('tconst').join(IMDB_akas.set_index('titleId')['title'], how='inner')

# Use key to merge title, rating, year of release
IMDB_ratings = IMDB_ratings.join(IMDB_basics.set_index('tconst'), how='inner')

# Put titles in lower case
df_movie['Movie name'] = df_movie['Movie name'].str.lower()
IMDB_ratings.title = IMDB_ratings.title.str.lower()

# Convert dates to year format and int type
def date_to_int(x):
    if (isinstance(x, str) and len(x) >= 4):
        return int(x[:4])
    else:
        return x

df_movie['Movie release year'] = df_movie['Movie release date'].apply(lambda x : date_to_int(x))
IMDB_ratings.startYear = IMDB_ratings.startYear.apply(lambda x : date_to_int(x))

# Merge ratings to our movie dataset (and drop duplicate columns)
df_movie = df_movie.join(IMDB_ratings.set_index(['title', 'startYear']),
                         how='left', on = ['Movie name', 'Movie release year'])

# Save expanded data
df_movie.to_csv('Expanded_data/movie.expanded_metadata.tsv', sep='\t', index = False)