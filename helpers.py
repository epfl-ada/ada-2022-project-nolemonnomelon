##################################################################
# This script contains some helper functions
##################################################################


import pandas as pd
import numpy as np
import ast

def transform_into_list(text) :
    ''' This function take a string and transform it into list of strings'''
    if isinstance(text, str) :
        if text != '[]' :
            text = text[1:-1]
            text = text.replace("'", "")
            return list(subString for subString in text.split(', '))
    else :
        return float('NaN')   
    

def incorporate_genre_dummies(data):
    ''' Add genres as dummy variables '''
    # transform into dummies
    movie_genres = [ast.literal_eval(movie_genre) for movie_genre in data['Movie genres names']]
    df = pd.get_dummies(pd.DataFrame(movie_genres))
    df.columns = df.columns.str.split("_").str[-1]

    # need to sum similarly named columns due to unwanted effect of previous computation
    df = df.groupby(level=0, axis=1).sum()
    genre_names = df.columns

    # adding to data and removing old genre column
    data[df.columns] = df.values
    #data = data.drop('Movie genres names', axis = 1)

    # rename problematic Sci-Fi column name
    data.rename(columns={'Sci-Fi' : 'SciFi'}, inplace = True)
    genre_names = [x if x != 'Sci-Fi' else 'SciFi' for x in genre_names]

    return data, genre_names
