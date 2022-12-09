#################################################################################
# This file query budget and origin country for movies we know the revenue for. #
#################################################################################

import requests
import pandas as pd
import json

# Import idx corresponding to known box office revenue
big_data = pd.read_csv('../Expanded_data/big_data.tsv', sep='\t')
revenue_idx = big_data['Movie box office revenue'].dropna().index.values
# Some revenue are only identified by IMDB or freebase id, so we take both
IMDB_id = big_data['IMDB_id'].iloc[revenue_idx]
FREEBASE_id = big_data['Freebase movie ID'].iloc[revenue_idx]
# reshaping
IMDB_id = [val for val in IMDB_id]
FREEBASE_id = [val for val in FREEBASE_id]

# Need API key --> https://developers.themoviedb.org/3/getting-started/authentication
# Créer un compte : https://www.themoviedb.org/login
# Loris API key : 0029604b00a495c511691a4a686ad4db
API_key = '0029604b00a495c511691a4a686ad4db'

# CREATE FUNCTIONS TO HANDLE QUERIES
def unpack_query(response):
    """ Unpacks raw query response into dictionnary-like structure """
    array = response.json()
    text = json.dumps(array)
    return json.loads(text)

def get_movie_id(API_key, imdb_id, freebase_id):
    """ Does the query given a imdb_id and returns movie id """
    # query choosing available source
    if imdb_id is not None and pd.notna(imdb_id):
        source = 'imdb_id'
        id_ = imdb_id
    else:
        source = 'freebase_id'
        id_ = freebase_id

    response = requests.get('https://api.themoviedb.org/3/find/' + id_ +
                                '?api_key=' + API_key + '&external_source=' + source)
    # successful query
    if response.status_code==200:
        # fetch movie id
        try:
            id_ = unpack_query(response)["movie_results"][0]['id']
            return id_
        except:
            return None
    else:
        return None

def extract_features(movie_id, budget, prod_country):
    """ Does the query for relevant features given the extracted movie id and stores them in lists """

    # create default values for budget and prod_country
    budget_ = None
    prod_country_ = None

    # query if possible
    if movie_id is not None:
        response = requests.get('https://api.themoviedb.org/3/movie/' + str(movie_id) +
                                 '?api_key=' + API_key + '&language=en-US')
        # successful query => try to get features
        if response.status_code==200:
            info = unpack_query(response)
            try:
                budget_ = info['budget'] if info['budget'] != 0 else None
            except: pass
            try:
                country = info['production_companies'][0]['origin_country']
                prod_country_ = country if country != '' else None
            except: pass
    budget.append(budget_)
    prod_country.append(prod_country_)

# Calls the function on revenue-corresponding-IDs
budget, prod_country = [], []
for i, (imdb_id, freebase_id) in enumerate(zip(IMDB_id, FREEBASE_id)):
    if i%1000 == 0: print(f'Query {i}/ {len(IMDB_id)} performed.')
    extract_features(get_movie_id(API_key, imdb_id, freebase_id), budget, prod_country)

# Create dataframe and stores data
print(len(IMDB_id), len(FREEBASE_id), len(budget), len(prod_country))
d = {'IMDB_id' : IMDB_id, 'Freebase_id' : FREEBASE_id, 'budget' : budget, 'prod_country' : prod_country}
TMDB_df = pd.DataFrame(data=d)
TMDB_df.to_csv('../Expanded_data/TMDB_query.tsv', sep='\t', index = False)