##############################################
# This script contains some helper functions #
##############################################

import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy.stats import pearsonr
from sklearn import decomposition
import itertools

import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
pio.renderers.default = 'jupyterlab'

import holoviews as hv
from holoviews import opts, dim
hv.extension('bokeh')
hv.output(size=200)

# hide unexplained holoview warning
import warnings
warnings.filterwarnings('ignore')

def incorporate_genre_dummies(data):
    """Add genres as dummy variables

    Args:
        data (dataframe): dataframe with column 'Movie genres names' to make dummmies from

    Returns:
        dataframe: same dataframe with dummies added
    """
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


def bootstrap(data, n_it, also_median = False):
    """ Computes a 95% CI around the data mean and median if necessary

    Args:
        data (list/np.array): data to bootstrap
        n_it (int): number of times to do a bootstrap approximation
        also_median (bool, optional): True if metrics for median should also be returned. Defaults to False.

    Returns:
        list: mean(median) and 95% interval percentiles
    """
    means = np.zeros(n_it)
    data = np.array(data)
    if also_median:
        medians = np.zeros(n_it)
    
    for n in range(n_it):
        indices = np.random.randint(0, len(data), len(data))
        data_new = data[indices] 
        means[n] = np.nanmean(data_new)
        if also_median:
            medians[n] = np.nanmedian(data_new)

    if also_median:
        return [np.nanmean(means), np.nanmedian(medians), np.nanpercentile(means, 2.5),np.nanpercentile(means, 97.5), np.nanpercentile(medians, 2.5),np.nanpercentile(medians, 97.5)]
    else:
        return [np.nanmean(means), np.nanpercentile(means, 2.5),np.nanpercentile(means, 97.5)]


def difference_in_usage(data_, g, CI_list, measure) :
    """ Computes bootstrap estimate of 95% CI for a given column in dataframe and appends it to a list

    Args:
        data_ (dataframe): dataframe containing the columns we want to bootstrap on
        g (string): column which contains 1 at elements which we will use for the bootstrap
        CI_list (list): list to which we append the bootstrap estimates
        measure (string): measure we bootstrap on (conditional on g)
    """
    #difference returns -1 (absent for winner and present for the loser), 0 (present or absent in both), or 1 (present for the winner and absent for looser). 
    d = data_[data_[g]==1][measure]
    b = bootstrap(d, 1000)
    CI_list.append(b)


def hist_subplots(data, measure, genre_names, cmap, subtitle, xlabel, xlim, ylim):
    """ Plots histograms of a given measure for each genre

    Args:
        data (dataframe): complete movie dataframe
        measure (string): measure we will count
        genre_names (list of strings): genres we will do an histogram for
        cmap (list of rgb colors): colormap associated to genre_names
        subtitle (string): title of plot
        xlabel (string): label of x axis
        xlim (int/tuple): limits on x axis
        ylim (int/tuple): limits on y axis
    """
    fig, axs = plt.subplots(5,4, constrained_layout=True, figsize=(20, 15))
    fig.suptitle(subtitle)

    for i, g in enumerate(genre_names):
        ax = axs[int(i/4),i%4]
        ax.hist(data[data[g]==1][measure], density=True, color = cmap[g], bins=30)
        ax.set_title(g)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set(xlabel=xlabel, ylabel='Density')


def plot_CIs(CIs, params, m_colors, xlabel=None, n=None, fig=None, axs=None):
    """Function to plot confidence intervals (originally adapted code from solutions of tutorial 4)

    Args:
        CIs (tuple list): list of lower and upper percentiles for each category
        params (list of strings): categories plotted
        m_colors (list of rgb colors): colormap associated to genre_names
        xlabel (string, optional): label of x. Defaults to None.
        n (int, optional): ax number to plot on. Defaults to None.
        fig (fig object, optional): figure containing the axes. Defaults to None.
        axs (ax object, optional): figure ax to plot the CIs on. Defaults to None.
    """

    # Compute interval center and half interval length for plotting
    means = np.array([CI[0] for CI in CIs])
    one_sided_CI = np.array([(CI[2] - CI[1]) / 2 for CI in CIs])
    Y = np.array(range(len(means)))
    
    if n == None:
        plt.figure(figsize=(8,5))
        for mean, y, c, err in zip(means, Y, m_colors, one_sided_CI):
            plt.errorbar(mean, y, xerr=err, linewidth=1, color = 'royalblue',
                        linestyle='none', marker='o', markersize=7,
                        markerfacecolor=c, markeredgecolor='black', capsize=5)
        #plt.vlines(0, 0, len(means), linestyle='--')
        plt.yticks(range(len(params)), params);
        plt.xlabel(xlabel)
        plt.title('Means 95% confidence intervals')
        plt.show()
    else:
        for mean, y, c, err in zip(means, Y, m_colors, one_sided_CI):
            axs[n].errorbar(mean, y, xerr=err, linewidth=1, color = 'royalblue',
                            linestyle='none', marker='o', markersize=7,
                            markerfacecolor=c, markeredgecolor='black', capsize=5)
        #plt.vlines(0, 0, len(means), linestyle='--')
        axs[n].set_yticks(range(len(params)), params);
        axs[n].set_xlabel(xlabel)
        fig.suptitle('Means 95% confidence intervals')

    
def plot_double_CIs(CIs_t1, CIs_t2, params, m_colors, xlabel=None, figsize=(10,8), n = None, fig=None, axs=None):
    """Function to plot confidence intervals (originally adapted code from solutions of tutorial 4)

    Args:
        CIs_t1 (tuple list): list of lower and upper percentiles for each category, group 1
        CIs_t2 (tuple list): list of lower and upper percentiles for each category, group 2
        params (list of strings): categories plotted
        m_colors (list of rgb colors): colormap associated to genre_names
        xlabel (string, optional): label of x. Defaults to None.
        figsize (tuple, optional): size of figure.
        n (int, optional): ax number to plot on. Defaults to None.
        fig (fig object, optional): figure containing the axes. Defaults to None.
        axs (ax object, optional): figure ax to plot the CIs on. Defaults to None.
    """

    # Compute interval center and half interval length for plotting for t1
    means_1 = np.array([CI[0] for CI in CIs_t1])
    one_sided_CI_1 = np.array([(CI[2] - CI[1]) / 2 for CI in CIs_t1])
    
    # Compute interval center and half interval length for plotting for t2
    means_2 = np.array([CI[0] for CI in CIs_t2])
    one_sided_CI_2 = np.array([(CI[2] - CI[1]) / 2 for CI in CIs_t2])

    # Create variable to space the CIs vertically
    Y = np.array(range(len(means_1)))

    if n == None:
        plt.figure(figsize=figsize)
        
        for mean, y, c, err in zip(means_1, Y, m_colors, one_sided_CI_1):
            # plot CIs
            plt.errorbar(mean, y + 0.2, xerr=err, linewidth=1, color = 'royalblue',
                    linestyle='none', marker='o', markersize=7,
                    markerfacecolor=c, markeredgecolor='black', capsize=5)

        for mean, y, c, err in zip(means_2, Y, m_colors, one_sided_CI_2):
            # plot CIs
            plt.errorbar(mean, y - 0.1, xerr=err, linewidth=1, color = 'darkorange',
                    linestyle='none', marker='o', markersize=7,
                    markerfacecolor=c, markeredgecolor='black', capsize=5)
    
        plt.yticks(range(len(params)), params);
        plt.xlabel(xlabel)
        plt.title('Means 95% confidence intervals')
        plt.legend([plt.plot([],ls="-", color='royalblue')[0],
                    plt.plot([],ls="-", color='darkorange')[0]],
                    ['old', 'recent'])
        

    else:

        for mean, y, c, err in zip(means_1, Y, m_colors, one_sided_CI_1):
            # plot CIs
            axs[n].errorbar(mean, y + 0.2, xerr=err, linewidth=1, color='royalblue',
                            linestyle='none', marker='o', markersize=7,
                            markerfacecolor=c, markeredgecolor='black', capsize=5)
        
        for mean, y, c, err in zip(means_2, Y, m_colors, one_sided_CI_2):
            # plot CIs
            axs[n].errorbar(mean, y - 0.1, xerr=err, linewidth=1, color='darkorange',
                            linestyle='none', marker='o', markersize=7,
                            markerfacecolor=c, markeredgecolor='black', capsize=5)
    
        axs[n].set_yticks(range(len(params)), params);
        axs[n].set_xlabel(xlabel)
        fig.suptitle('Means 95% confidence intervals')
        axs[n].legend([plt.plot([],ls="-", color='royalblue')[0],
                    plt.plot([],ls="-", color='darkorange')[0]],
                    ['old', 'recent'])


def barplot(res, xlabel, significant_only = False, cmap = None, figsize=(5,7)) :
    """Barplot of the coefficients of the linear regression sorted by value

    Args:
        res (mod.fit() object): results from model fitted usinf smf.ols()
        xlabel (string): label of x
        significant_only (bool, optional): True if we want to plot only the significant coefficients. Defaults to False.
        cmap (dict of rgb colors, optional): Colormap corresponding to the fitted parameters. Defaults to None.
        figsize (tuple, optional): size of the figure. Defaults to (5,7).
    """
    tmp = []
    for name, value, p_val in zip(res.params.index, res.params, res.pvalues):
        if not significant_only:
            add_to_title = ''
            tmp.append({"name": name, "value": value})
        else:
            add_to_title = 'significant'
            if p_val < 0.05 / len(res.params):
                tmp.append({"name": name, "value": value})

    features_coef = pd.DataFrame(tmp).sort_values("value")

    plt.subplots(figsize=figsize)
    cmap['Intercept'] = 'k'
    plt.barh(features_coef.name, features_coef.value,
            color = [cmap[genre] for genre in features_coef.name],
            alpha=0.6)
    plt.title(f'Regression coefficients of {add_to_title} genres')
    plt.xlabel(xlabel)
    plt.show()


###################
### Exploratory ###
###################


def mean_median(data, feature):
    """ Computes mean and median for a the wanted features by years

    Args:
        data (dataframe): contains the features
        feature (list): contains 2 features for which we compute the metrics

    Returns:
        list: years in data, mean f1, median f1, mean f2, median f2
    """
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


def plot_RRB_distr(dist, log=[False, False], xlim=True, title = 'Histrogram distribution', filename = None, save=False):
    """ Plots histogram of our revenue, rating, budget distributions

    Args:
        dist (dataframe): contains the features
        log (list, optional): possibility to logscale the axes. Defaults to [False, False].
        xlim (bool, optional): possibility the limit the x axe. Defaults to True.
        title (str, optional): plot title. Defaults to 'Histrogram distribution'.
        filename (str, optional): filename in case saving is enabled. Defaults to None.
        save (bool, optional): possibility to save plot. Defaults to False.
    """
    fig, axs = plt.subplots(3, 1, figsize=(16, 14), sharey=True)
    sns.histplot(dist.averageRating, ax=axs[0], binwidth=0.1)
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

def plot_RRB_across_time(years, mean, median, low, high, low_median, high_median, filename = None, save = False):
    """ Plots mean and median revenue, rating, budget through time with CIs

    Args:
        years (np array): release years in dataset
        mean (np array): means per feature (3 rows)
        median (np array): medians per feature (3 rows)
        low (np array): lower CI percentiles for means per feature per year (3 rows)
        high (np array): upper CI percentiles for means per feature per year (3 rows)
        low_median (np array): lower CI percentiles for medians per feature per year (3 rows)
        high_median (np array): upper CI percentiles for medians per feature per year (3 rows)
        filename (str, optional): filename in case saving is enabled. Defaults to None.
        save (bool, optional): possibility to save plot. Defaults to False.
    """
    fig, ax = plt.subplots(3,1,figsize=(16,10),sharex=True)

    for i in range(3):
        ax[i].fill_between(years, low[i,:],high[i,:], alpha = 0.2, color = 'blue')
        l0, = ax[i].plot(years,mean[i,:], color = 'blue')

        ax[i].fill_between(years, low_median[i,:],high_median[i,:], alpha = 0.2, color = 'orange')
        l1, = ax[i].plot(years,median[i,:], color = 'orange')
        if i !=2 :
            ax[i].set_yscale('log')

    ax[0].set_ylabel('Revenue')
    ax[1].set_ylabel('Budget')
    ax[2].set_ylabel('Rating')
    plt.xlabel('Year')
    plt.suptitle('Mean and median of revenue, budget, rating from 1959 to 2021')
    fig.legend([l0,l1], ['Mean', 'Median'])
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


def reg_coef(x, y, color = None, label = None):
    """ Computes regression coefficients between x and y variables

    Args:
        x (np array): first variable
        y (np array): second variable
        color (None, optional): needed to use within scattering function. Defaults to None.
        label (None, optional): needed to use within scattering function. Defaults to None.
    """
    ax = plt.gca()
    r, p = pearsonr(x, y)
    ax.annotate('r = {:.2f}'.format(r), xy=(0.5, 0.6),
                xycoords='axes fraction', ha='center')
    ax.annotate('p = {:.2f}'.format(p), xy=(0.5, 0.5),
                xycoords='axes fraction', ha='center')
    ax.set_axis_off()


def scattering(data, title, Type = None, color = None, add_kde = False, filename = None, save = False):
    """ Scatters relations between our RRB variables

    Args:
        data (dataframe): contains the variables
        title (string): plot title
        Type (string, optional): Either median or mean. Defaults to None.
        color (string, optional): plot color. Defaults to None.
        add_kde (bool, optional): Possibility to draw density on top of the scattering. Defaults to False.
        filename (str, optional): filename in case saving is enabled. Defaults to None.
        save (bool, optional): possibility to save plot. Defaults to False.
    """
    if Type != 'All data':
        g = sns.PairGrid(data.loc[data['Type'] ==
                     Type], hue='Type', palette=[color])
    else:
        g = sns.PairGrid(data)
    g.map_diag(sns.histplot, kde=True)
    g.map_lower(sns.regplot)
    if add_kde:
        g.map_lower(sns.kdeplot, color = 'red')
    g.map_upper(reg_coef)
    g.fig.suptitle(title)
    if save:
        plt.savefig(f'outputs/{filename}.png')
    plt.show()


###################
###    Genre    ###
###################

def year_distribution(data, title, filename = None, save=False):
    """ Plots the number of movies per year in data using plotly

    Args:
        data (dataframe): contains movies and their release date
        title (string): title of plot
        filename (string, optional): name of file if saving is enabled. Defaults to None.
        save (bool, optional): option to save the plot. Defaults to False.
    """
    df_column = data.copy()
    df_column.dropna(subset=['Movie release date'], inplace=True)
    count = df_column['Movie release date'].value_counts()
    # plt.figure(figsize=(20,8))
    df = pd.DataFrame({'Year': count.index.astype('int64'), 'Movie count': count.values})
    fig = px.bar(df, x='Year', y='Movie count',
                 color='Movie count', title=title)
    fig.update_layout(title_x=0.5)
    if save:
        fig.write_html(f"outputs/{filename}.html")
    fig.show('png')

def hbarplot(x, y, title, colors = 'Blues_r', filename = None, save = False):
    """ Plots the number of movies per year in data using matplotlib

    Args:
        x (list): x variables
        y (list): y variables
        title (string): plot title
        colors (str/list, optional): list of rgb colors or colormap associated to the y variables. Defaults to 'Blues_r'.
        filename (string, optional): name of file if saving is enabled. Defaults to None.
        save (bool, optional): option to save the plot. Defaults to False.
    """
    plt.figure(figsize=(15, 8))
    sns.barplot(x=x, y=y, palette=colors).set(title=title)
    if save:
        plt.savefig(f'outputs/{filename}.png')
    plt.show()


def plotly_barplot(df, x, y, cmap, title = 'Barchart', err_bar = False, subplots=None, filename = None, save = False):
    """ Plots the x-values associated to the y variable (movie genres) in df

    Args:
        df (dataframe): contains the data
        x (string): df column containing the variable associated to genres
        y (string): column containing the y variable (movie genre)
        cmap (dict): dict of hex colors associated to the y variables
        title (str, optional): plot title. Defaults to 'Barchart'.
        err_bar (bool, optional): error bars associated to the x variables. Defaults to False.
        subplots (string, optional): Dataframe column for which subplots will be made. Defaults to None.
        filename (string, optional): name of file if saving is enabled. Defaults to None.
        save (bool, optional): option to save the plot. Defaults to False.
    """
    fig = px.bar(df, x=x, y=y, color = y, title=title, facet_col=subplots,
                    color_discrete_map=cmap)
    
    if err_bar:
        for med, genre, low, high in zip(df[x], df[y], df['low'], df['high']):
            fig.add_trace(go.Scatter(
                x=[med],
                y=[genre],
                error_x=dict(type='data',color = 'black',array=[med-low, high-med],visible=True),
                marker=dict(color='rgba(0,0,0,0)', size=12),
                showlegend=False
                ))

    fig['layout']['showlegend'] = False

    fig.update_layout(title_x=0.5, autosize=False, width=800,
            height=600, showlegend=False, yaxis={'categoryorder':'total ascending'})
    # fig['layout']['yaxis']['autorange'] = "reversed"
    if save:
        fig.write_html(f"outputs/{filename}.html")
    fig.show('png')
    

def cooccurence_matrix(movie_genres, genre_list):
    """ Creates connectivity matrix from movie genre lists

    Args:
        movie_genres (list of lists): associations of genres for every movie
        genre_list (list): individual genres

    Returns:
        np array, dict: co-occurence matrix, genre to index mapping associated to the matrix
    """
    cooc_matrix = np.zeros((len(genre_list), len(genre_list))).astype(int)
    # create index to genre mapping
    map_ = {genre: i for i, genre in enumerate(genre_list)}

    # we create a connection for all pairwise genre combination and a count individual occurences
    for genre_list in movie_genres:
        # handle unique genre
        if len(genre_list) == 1:
            genre = genre_list[0]
            cooc_matrix[map_[genre], map_[genre]] += 1

        # count all pairwise combinations and count individual occurences
        else:
            for comb in itertools.combinations(genre_list, 2):
                # increment on both sides of symmetric matrix
                cooc_matrix[map_[comb[0]], map_[comb[1]]] += 1
                cooc_matrix[map_[comb[1]], map_[comb[0]]] += 1

    # delete symmetric values but keep diagonal
    cooc_matrix = np.triu(cooc_matrix, k=0)
    return cooc_matrix, map_

def prepare_chord(cooc_matrix, map_):
    """ Transform the connectivity matrix into edges and nodes for holoview formats

    Args:
        cooc_matrix (np array): coocurrence matrix for movie genres
        map_ (dict): dict: genre to index mapping associated to the matrix

    Returns:
        dataframe, holoview Dataset object: cooccurence in holoview readable format
    """
    # Build a dataframe with connections
    edge_list = pd.DataFrame(cooc_matrix).stack()
    edge_list = edge_list[edge_list != 0]

    # prepare format for holoview
    links = pd.DataFrame({'source' : [ind[0] for ind in edge_list.index],
                        'target' : [ind[1] for ind in edge_list.index],
                        'value' : edge_list.values})
    nodes = hv.Dataset(pd.DataFrame(
        {'Genre': map_.keys(), 'group': map_.values()}), 'index')

    return links, nodes

def one_chord(links, nodes, thres, colors):
    """ Create a chord diagram

    Args:
        links (dataframe): source targets and value (cooccurence)
        nodes (holoview dataset object): index to genre mapping
        thres (int): min number of co-occurences
        colors (dict): colormap associated to genres

    Returns:
        holoview diagram object: one diagram frame
    """
    # drop elements under threshold
    links.drop(links[links['value'] < thres].index, inplace=True)
    # obtain remaining nodes
    new_idx = np.union1d(links.source.unique(), links.target.unique())
    nodes.data.drop(nodes.data[~nodes.data['index'].isin(new_idx)].index, inplace=True)
    
    chord = hv.Chord((links, nodes)).select(value=(0, None))
    diag = chord.opts(
            opts.Chord(cmap=[colors[genre] for genre in nodes['Genre']],
                    edge_cmap=[colors[genre] for genre in nodes['Genre']],
                    edge_color=dim('source').str(), 
                    labels='Genre', node_color=dim('Genre').str()))
    return diag

def multiple_chord(links, nodes, thres_list, colors, save = False):
    """ Creates multiple diagram frames for differents thresholds by calling one_chord()

    Args:
        links (dataframe): source targets and value (cooccurence)
        nodes (holoview dataset object): index to genre mapping
        thres (int): min number of co-occurences
        colors (dict): colormap associated to genres
        save (bool, optional): option to save the plot. Defaults to False.

    Returns:
        aggregated diagram object: all frames aggregated
    """
    dict_ = {}
    for t in thres_list:
        dict_[f'{t}'] = one_chord(links, nodes, t, colors)
    hmap = hv.HoloMap(dict_, 'Min co-occurences').opts(fontsize={'title': 16})
    hv.util.save(hmap, 'outputs/genre_chord_diag', fmt='html', resources='cdn', toolbar=True, title='Genre co-occurences')
    plt.show('png')

########################
###    Clustering    ###
########################    

def four_radar_charts(data_R_B, whole_data, name_cat='', ran=[0, 0.55]) :  
    """ Plot 4 radar charts from the data_R_B. We computed for each feature the fraction of
    of this feature with respect to the whole data set.

    Args:
        data_R_B (list of dataframes): contains 4 subsets
        whole_data (dataframe): whole dataset
        name_cat (str, optional): name of the category (genre/production country) analysed. Defaults to ''.
        ran (list, optional): limits of the plot radius scale. Defaults to [0, 0.55].
    """
    # Columns names to label the charts, the same for all the 4 data sets
    categories = data_R_B[0].columns.values

    fig = make_subplots(rows=2, cols=2, specs=[[{'type': 'polar'}]*2]*2)

    # Total count of the genres to normalize the count of the 4 groups 
    total_count = whole_data.sum().values

    fig.append_trace(go.Scatterpolar(r=data_R_B[0].sum().values/total_count, theta=categories, fill='toself', name='High rating - Low budget'), row=1, col=1)
    fig.append_trace(go.Scatterpolar(r=data_R_B[1].sum().values/total_count, theta=categories, fill='toself', name='High rating - High budget'), row=1, col=2)
    fig.append_trace(go.Scatterpolar(r=data_R_B[2].sum().values/total_count, theta=categories, fill='toself', name='Low rating - High budget'), row=2, col=1)
    fig.append_trace(go.Scatterpolar(r=data_R_B[3].sum().values/total_count, theta=categories, fill='toself', name='Low rating - Low budget'), row=2, col=2)

    fig.update_polars(radialaxis=dict(range=ran))

    fig.update_layout(
      polar=dict(
        radialaxis=dict(
          visible=True,
        )),
        width=1300,
        height=1200,
      showlegend=True,
        title_text = '{} representation in the subsets'.format(name_cat)
    )

    fig.show('jupyterlab')
    
def four_radar_charts_superposition(data_R_B, whole_data, name_cat='', ran=[0, 0.55]) :
    """ Plot 4 radar charts from the data_R_B, each subset being superimposed. We computed for each feature the fraction of
    of this feature with respect to the whole data set.

    Args:
        data_R_B (list of dataframes): contains 4 subsets
        whole_data (dataframe): whole dataset
        name_cat (str, optional): name of the category (genre/production country) analysed. Defaults to ''.
        ran (list, optional): limits of the plot radius scale. Defaults to [0, 0.55].
    """
    # Columns names to label the charts, the same for all the 4 data sets
    categories = data_R_B[0].columns.values
    # Total count of the genres to normalize the count of the 4 groups 
    total_count = whole_data.sum().values
    
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
          r=data_R_B[0].sum().values/total_count,
          theta=categories,
          fill='toself',
          name='High rating, low budget'
    ))
    fig.add_trace(go.Scatterpolar(
          r=data_R_B[1].sum().values/total_count,
          theta=categories,
          fill='toself',
          name='High rating, high budget'
    ))

    fig.add_trace(go.Scatterpolar(
          r=data_R_B[2].sum().values/total_count,
          theta=categories,
          fill='toself',
          name='Low rating, high budget'
    ))

    fig.add_trace(go.Scatterpolar(
          r=data_R_B[3].sum().values/total_count,
          theta=categories,
          fill='toself',
          name='Low rating, low budget'
    ))

    fig.update_layout(
      polar=dict(
        radialaxis=dict(
          visible=True,
          range=ran
        )),
        width=1300,
        height=1000,
      showlegend=True,
        title_text = '{} representation in the subsets'.format(name_cat)
    )

    fig.show('jupyterlab')
    
def compute_ttest(data_1, data_vs, feat_list, data_1_name='High rating - low budget', data_vs_name= 'the rest') : 
    """ Compute independent t-tests of the data_1 vs data_vs dataframes the given features. Also prints the results

    Args:
        data_1 (dataframe): first subset
        data_vs (dataframe): second subset
        feat_list (list): features to analyse
        data_1_name (str, optional): name of the first subset. Defaults to 'High rating - low budget'.
        data_vs_name (str, optional): name of the second subset. Defaults to 'the rest'.
    """
    for f in feat_list : 
        print(f)
        print('{} vs {} pval : {}'.format(data_1_name, data_vs_name, \
            scipy.stats.ttest_ind(data_1[f], data_vs[f], alternative='greater').pvalue))
        print('\n')
             
def compute_and_plot_CI_with_data_separation(data_1, data_vs, m_colors, data_1_name = 'High rating - Low budget movies', data_vs_name = 'The rest of the data', actors_params=['mean_age', 'mean_height', 'fraction_men'], cutoff_date=2000) :
    """ Compute and plot CI with bootstrap from 2 different subsets (old and recent movies) for the actor attributes features (actors_params)

    Args:
        data_1 (dataframe): first subset
        data_vs (dataframe): second subset
        m_colors (_type_): colormap for CI markers (not really used in this function, used for genre analysis)
        data_1_name (str, optional): name of the first subset. Defaults to 'High rating - Low budget movies'.
        data_vs_name (str, optional): name of the second subset. Defaults to 'The rest of the data'.
        actors_params (list, optional): parameters to plot. Defaults to ['mean_age', 'mean_height', 'fraction_men'].
        cutoff_date (int, optional): date separating the two subsets. Defaults to 2000.
    """
    
    # Split data in two periods: from 1959 to 2000 and from 2000 to 2021.
    old_movies_1 = data_1[data_1.date < cutoff_date]
    recent_movies_1 = data_1[data_1.date >= cutoff_date]
    old_movie_data_vs = data_vs[data_vs.date < cutoff_date]
    recent_movie_data_vs = data_vs[data_vs.date > cutoff_date]
    
    CIs_old = []
    CIs_recent = []
    CIs_old_vs = []
    CIs_recent_vs = []
    
    # Boostrap for all the actors parameters and the data sets
    for p in actors_params :
        CIs_old.append(bootstrap(old_movies_1[p], 1000))
        CIs_recent.append(bootstrap(recent_movies_1[p], 1000))
        CIs_old_vs.append(bootstrap(old_movie_data_vs[p], 1000))
        CIs_recent_vs.append(bootstrap(recent_movie_data_vs[p], 1000))
    
    # Plot the CI
    for i in range(3) :
        plot_double_CIs([CIs_old[i], CIs_old_vs[i]], [CIs_recent[i], CIs_recent_vs[i]], params= \
                        [data_1_name, data_vs_name], xlabel=actors_params[i], figsize=(5,2), m_colors=m_colors)
        
def single_radar_chart(subset, whole_data, name_cat='') :   
    """ Plot a single radar chart of the subset data. We computed for each feature the fraction of
    of this feature with respect to the whole data set.

    Args:
        subset (dataframe): data to plot
        whole_data (dataframe): whole dataframe, used to normalize the results
        name_cat (str, optional): category of features plotted (genre, ...). Defaults to ''.
    """
    # Total count of the genres to normalize the count 
    total_count = whole_data.sum().values

    fig = go.Figure(data=go.Scatterpolar(r=subset.sum().values/total_count, theta=subset.columns.values, \
                                         fill='toself', name='high fraction (revenue / budget) and low budget'))

    fig.update_polars(radialaxis=dict(range=[0, 0.55]))

    fig.update_layout(
      polar=dict(
        radialaxis=dict(
          visible=True,
        )),
        width=950,
        height=500,
      showlegend=True,
        title_text = '{} representation in the subset'.format(name_cat)
        #title_text = 'Number of the movies over the total number of movies in the {} categorie in the whole dataset'.format(name_cat)
    )

    fig.show('jupyterlab')
    
def standardize(data) :
    """ Standardize data

    Args:
        data (np array): data to standardize

    Returns:
        np array: standardized data
    """
    return (data - np.nanmean(data)) / np.std(data)

def data_for_pca(data, not_split=True) :
    """Prepare the data for the PCA. Standardizes some features and drop one that is useless. If the data are split into train and test set
    (not_split=False), we want to only standardize the actor attributes

    Args:
        data (dataframe): data to prepare
        not_split (bool, optional): False if we split the data into train and test sets. Defaults to True.

    Returns:
        _type_: _description_
    """
    if not_split :
        data_pca = data.drop(columns=['log_fraction_rev_bud'])
        data_pca['log_budget'] = standardize(data_pca['log_budget'])
        data_pca['log_revenue'] = standardize(data_pca['log_revenue'])
        data_pca['averageRating'] = standardize(data_pca['averageRating'])
    else : 
        data_pca = data.copy()
    data_pca['mean_age'] = standardize(data_pca['mean_age'])
    data_pca['mean_height'] = standardize(data_pca['mean_height'])
    data_pca['fraction_men'] = standardize(data_pca['fraction_men'])
    return data_pca

def plot_PCA(data_pca, color_label='averageRating', name_columns = None, color_feat = None, n_components=10) :
    """ Compute and plot a PCA with n_components

    Args:
        data_pca (dataframe): data to do a pca through
        color_label (str, optional): specific color for the plot, either a column name or a given feature. Defaults to 'averageRating'.
        name_columns (str, optional): columns to do the PCA through. Defaults to None (which means all columns are taken).
        color_feat (list, optional): color list used for color coding if specified. Defaults to None.
        n_components (int, optional): number of PC to plot. Defaults to 10.
    """
    if color_feat is None : 
        color_feat = data_pca[color_label]
        
    pca = decomposition.PCA(n_components=n_components)
    if name_columns is None:
        components = pca.fit_transform(data_pca)
    else: components = pca.fit_transform(data_pca[name_columns])

    total_var = pca.explained_variance_ratio_.sum() * 100

    labels = {str(i): f"PC {i+1}" for i in range(n_components)}
    labels['color'] = 'standardized ' + color_label
    
    fig = px.scatter_matrix(
        components,
        color = color_feat,
        dimensions=range(n_components),
        labels=labels,
        title=f'Total Explained Variance: {total_var:.2f}%',
    )
    fig.update_traces(diagonal_visible=False)

    fig.update_layout(
        width=1300,
        height=1300,
        showlegend=True,
        )

    fig.show('jupyterlab')
    
def variance_explained_plot(pca) :
    """ Plot the PCs vs the total variance explained

    Args:
        pca (pca object): result of the pca fitting function
    """
    plt.figure(figsize=(8,5))
    plt.title('Cumulative variance explained')
    sns.lineplot(x = np.arange(1,11,1), y = np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Principal Components')
    plt.ylabel('Variance explained')
    plt.show()
    
def heatmap_pca(pca, index) :
    """ Plots the heatmap of the loads of the features associated with each PC

    Args:
        pca (pca object): result of the pca fitting function
        index (list): columns of the pca object
    """
    loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10'], index = index)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(loadings, annot=True, fmt='.2f')
    plt.show()

def split_set(data_to_split, ratio=0.7):
    """ Split the data into train and test sets

    Args:
        data_to_split (dataframe): data to split
        ratio (float, optional): percentage of training set. Defaults to 0.7.

    Returns:
        list: train set and test set
    """
    mask = np.random.rand(len(data_to_split)) < ratio
    return [data_to_split[mask].reset_index(drop=True), data_to_split[~mask].reset_index(drop=True)]