##################################################################
# This script contains some helper functions
##################################################################


import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt

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


def bootstrap(data, n_it):
    '''Bootstrap to get the 95% CI'''
    means = np.zeros(n_it)
    data = np.array(data)
    
    for n in range(n_it):
        indices = np.random.randint(0, len(data), len(data))
        data_new = data[indices] 
        means[n] = np.nanmean(data_new)
    
    # 95% CI -> 2.5% and 97.5%
    return [np.nanmean(means), np.nanpercentile(means, 2.5),np.nanpercentile(means, 97.5)]


def plot_CIs(CIs, params, xlabel=None):
    ''' Function to plot confidence intervals adapted code from solutions of tutorial 4 '''

    # Compute interval center and half interval length for plotting
    means = np.array([CI[0] for CI in CIs])
    one_sided_CI = np.array([(CI[2] - CI[1]) / 2 for CI in CIs])

    # plot CIs
    plt.figure(figsize=(8,5))
    plt.errorbar(means, np.array(range(len(means))), xerr=one_sided_CI, linewidth=1,
                 linestyle='none', marker='o', markersize=3,
                 markerfacecolor='black', markeredgecolor='black', capsize=5)
    #plt.vlines(0, 0, len(means), linestyle='--')
    plt.yticks(range(len(params)), params);
    plt.xlabel(xlabel)
    plt.title('95% confidence intervals')
    plt.show()

    
def plot_double_CIs(CIs_t1, CIs_t2, params, xlabel=None, figsize=(10,8)):
    # function to plot confidence intervals
    # adapted code from solutions of tutorial 4

    # create figure
    plt.figure(figsize=figsize)

    # Compute interval center and half interval length for plotting for t1
    means = np.array([CI[0] for CI in CIs_t1])
    one_sided_CI = np.array([(CI[2] - CI[1]) / 2 for CI in CIs_t1])

    # plot CIs
    l1 = plt.errorbar(means, np.array(range(len(means))) + 0.2, xerr=one_sided_CI, linewidth=1,
                 linestyle='none', marker='o', markersize=3,
                 markerfacecolor='black', markeredgecolor='black', capsize=5)

    # Compute interval center and half interval length for plotting for t2
    means = np.array([CI[0] for CI in CIs_t2])
    one_sided_CI = np.array([(CI[2] - CI[1]) / 2 for CI in CIs_t2])

    # plot CIs
    l2 = plt.errorbar(means, np.array(range(len(means))) - 0.1, xerr=one_sided_CI, linewidth=1,
                 linestyle='none', marker='o', markersize=3,
                 markerfacecolor='black', markeredgecolor='black', capsize=5)
    
    plt.yticks(range(len(params)), params);
    plt.xlabel(xlabel)
    plt.title('95% confidence intervals')
    plt.legend([l1, l2], ['old', 'recent'])


