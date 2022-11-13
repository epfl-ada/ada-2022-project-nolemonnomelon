#####################################
#  Merge TMDB query to big dataset  #
#####################################

import pandas as pd

# import dataset and query
big_data = pd.read_csv('Expanded_data/big_data.tsv', sep='\t')
TMDB = pd.read_csv('Expanded_data/TMDB_query.tsv', sep='\t')

# localize elements to merge by imdb or freebase id
imdb_idx = TMDB['IMDB_id'].dropna().index.values
freebase_idx = TMDB['Freebase_id'].dropna().index.values

# merge budget and production country of origin to big dataset
big_data = big_data.set_index('IMDB_id').join(TMDB.iloc[imdb_idx].set_index('IMDB_id')[['budget', 'prod_country']], how='left').reset_index()
big_data = big_data.set_index('Freebase movie ID').combine_first(TMDB.iloc[freebase_idx].set_index('Freebase_id')[['budget', 'prod_country']]).reset_index()

# reset index
big_data.rename(columns={'index' : 'Freebase movie ID'}, inplace = True)

# ADD ANY PREPROCESSING OF BIG DATA HERE ?

# Save expanded IMDB_wiki
big_data.to_csv('Expanded_data/big_data.tsv', sep='\t', index = False)

# AFTER THIS, LOAD BIG_DATA IN ANOTHER SCRIPT AND FINALLY COMPLETE CMU ?