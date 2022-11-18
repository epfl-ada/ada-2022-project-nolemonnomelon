#####################################
#  Merge TMDB query to big dataset  #
#####################################

import pandas as pd
import numpy as np
from helpers import transform_into_list

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


################################ PREPROCESSING ##########################################
# As the Movie release date is not homogeneous across all movies, we decided to only keep the year as a timestamp.
big_data['Movie release date'] = pd.to_datetime(big_data['Movie release date'], errors='coerce').dt.year.astype('Int64')

# We noticed that there are either 'NaN' or '\\N' for missing values in this dataset. So, we changed the '\\N' into 'NaN' for more consistency across the dataset
big_data.replace('\\N', float('NaN'), inplace=True)

# As the Movie genres are string, we transform it into list of strings
big_data['Movie genres names'] = big_data['Movie genres names'].apply(lambda s : transform_into_list(s)) 

# Replace the \\N by NaN
def delete_N(g) :
    ''' This function replace list of \\N by NaN. We do not understand why it is \\\\N to verify ; when we printed in the value it is \\N only... 
    However, it functions like this. '''
    if isinstance(g, list) :
        if g[0] == '\\\\N'  :
            return float('nan')
    return g
big_data['Movie genres names'] = big_data['Movie genres names'].apply(lambda x : delete_N(x))


################################ INFLATION ##########################################
# Inflation per country from 1960 to 2021
df_inflation = pd.read_csv("Inflation_data/inflation.csv",header=2)
df_inflation = df_inflation.iloc[:,0:-1]

# Inflation in the USA from 1960 to 2021 as the revenue and budget are calculated in US dollars
usa_inflation = df_inflation[df_inflation['Country Code'] == 'USA']

# Create the inflation coefficients per year 
inflation_coef = np.zeros(63)
inflation_coef[0] = 1
for i in range(62):
    year = i + 1960
    inflation_coef[i+1] = inflation_coef[i] * (1 + usa_inflation[str(year)]/100)   
    
# Function that gets the coefficient coeff according to the year
def get_inflation(year):
    if isinstance(year, int) :
        x = year-1959
        if x >= 0 and x < len(inflation_coef) :
            return inflation_coef[-1]/inflation_coef[x]
    return float('nan')
# Add the inflation coeff column to the dataframe 
big_data["inflation coeff"] = big_data["Movie release date"].apply(get_inflation)
# Corrected Movie box office revenue by inflation
big_data["inflation corrected revenue"] = big_data['Movie box office revenue'] * big_data["Movie release date"].apply(get_inflation)
# Corrected Movie budget by inflation
big_data["inflation corrected budget"] = big_data['budget'] * big_data["Movie release date"].apply(get_inflation)


################################ SAVE DATA ##########################################
# Save expanded IMDB_wiki
big_data.to_csv('Expanded_data/big_data_final.tsv', sep='\t', index = False)
