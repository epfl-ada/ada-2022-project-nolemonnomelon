##################################################################
# This script contains some helper functions
##################################################################


import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

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


###################
### Exploratory ###
###################


def mean_median(data, feature):
    years = np.sort(data['Movie release date'].unique())
    mean_without_inf = np.zeros_like(years)
    mean_with_inf = np.zeros_like(years)
    median_without_inf = np.zeros_like(years)
    median_with_inf = np.zeros_like(years)

    for i, y in enumerate(years):

        without_inf = data.loc[data['Movie release date'] == y][feature[0]]
        with_inf = data.loc[data['Movie release date'] == y][feature[1]]

        mean = np.mean(without_inf)
        median = np.median(without_inf)
        mean_without_inf[i] = mean
        median_without_inf[i] = median

        mean = np.mean(with_inf)
        median = np.median(with_inf)
        mean_with_inf[i] = mean
        median_with_inf[i] = median

    return [years, mean_without_inf, median_without_inf, mean_with_inf, median_with_inf]

def year_distribution(data, title, filename = None, save=False):
    df_column = data.copy()
    df_column.dropna(subset=['Movie release date'], inplace=True)
    count = df_column['Movie release date'].value_counts()
    # plt.figure(figsize=(20,8))
    df = pd.DataFrame({'Year': count.index.astype(
        'int64'), 'Movie count': count.values})
    fig = px.bar(df, x='Year', y='Movie count',
                 color='Movie count', title=title)
    fig.update_layout(title_x=0.5)
    if save:
        fig.write_html(f"outputs/{filename}.html")
    fig.show('jupyterlab')

# Plotting


def plot_RRB_distr(dist, log=[False, False], xlim=True, title = 'Histrogram distribution', filename = None, save=False):
    fig, axs = plt.subplots(3, 1, figsize=(16, 14), sharey=True)
    sns.histplot(dist.averageRating, ax=axs[0])
    sns.histplot(dist['inflation corrected revenue'], ax=axs[1], log_scale=log)
    if xlim == True:
        axs[1].set_xlim([0, 8e8])
    sns.histplot(dist['inflation corrected budget'], ax=axs[2], log_scale=log)
    fig.suptitle(title)
    axs[0].set_xlabel('Rating')
    axs[1].set_xlabel('Revenue')
    axs[2].set_xlabel('Budget')
    if save:
        plt.savefig(f'outputs/{filename}.png')
    plt.show()


def plot_mean_median(input_mean_median, ylabel, filename = None, save=False):
    fig, ax = plt.subplots(2, 1, figsize=(16, 10), sharey=True)

    ax[0].plot(input_mean_median[0], input_mean_median[1],
               color='blue', label='Mean')
    ax[0].plot(input_mean_median[0], input_mean_median[2],
               color='orange', label='Median')

    ax[1].plot(input_mean_median[0], input_mean_median[3], color='blue')
    ax[1].plot(input_mean_median[0], input_mean_median[4], color='orange')

    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    ax[0].set_ylabel(ylabel[0])
    ax[1].set_ylabel(ylabel[1])
    ax[0].set_xlabel('Year')
    ax[1].set_xlabel('Year')
    plt.suptitle('Mean and median ' +
                 ylabel[0] + ' and ' + ylabel[1] + ' from 1959 to 2021')
    fig.legend()
    if save:
        plt.savefig(f'outputs/{filename}.png')
    plt.show()


def reg_coef(x, y, label=None, color=None, **kwargs):
    ax = plt.gca()
    r, p = pearsonr(x, y)
    ax.annotate('r = {:.2f}'.format(r), xy=(0.5, 0.6),
                xycoords='axes fraction', ha='center')
    ax.annotate('p = {:.2f}'.format(p), xy=(0.5, 0.5),
                xycoords='axes fraction', ha='center')
    ax.set_axis_off()


def scattering(data, Type = None, color = None, filename = None, save = False):
    if Type != 'All data':
        g = sns.PairGrid(data.loc[df_pdataair['Type'] ==
                     Type], hue='Type', palette=[color])
    else:
        g = sns.PairGrid(data)
    g.map_diag(sns.histplot, kde=True)
    g.map_lower(sns.regplot)
    g.map_upper(reg_coef)
    g.fig.suptitle(Type)
    if save:
        plt.savefig(f'outputs/{filename}.png')
    plt.show()
