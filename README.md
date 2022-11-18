# How to make a good movie without money

___Study of the evolution of movie success across time___

## Abstract

<p align="justify"> In the late 19th century the entertainment industry saw the advent of the first films. In the past hundred years movies have become an integral part of the lives of people around the globe. Nowadays, the movie industry has seen the advent of "superproductions" supported by insane amounts of money. However, budget might not be a mandatory prerequisite for movie quality, appreciation and revenue. We aim to study the relationship between movie rating, revenue, and budget throughout the last 60 years. In addition, we want to include movie genres in our analysis, and hope to reveal a time-dependent effect of trends in the movie industry. This would be signalled by an increased revenue/rating due only to genre differences, changing across decades. Lastly, we are interested in low budget movies with high revenue and ratings. We want to investigate possible common characteristics of these movies and how they evolve with time. </p>



## Research Questions

Perform a temporal analysis of the changes of the popular movie genres.

1) Is there a link between revenue, budget and rating? Across time and in general.
2) Which genre is the cheapest to bring on screen ? Which genre is the most appreciated ? Across time and in general.
3) Can we find a cluster of good movies that have a low budget ? If yes, which percentage ? Which are the common features across them ?

## Proposed additional datasets
<p align="justify"> The CMU movie metadata contains not many and not recent movies (until 2012 only). Moreover, it has a lot of NA values, espcially for the box office revenue. Therefore, we decided to complete this dataset in order to have a more representative one. We use : </p>

*	[IMDB](https://datasets.imdbws.com/): dataset to complete the amount of movies of the original [CMU](http://www.cs.cmu.edu/~ark/personas/) and ensure good representation of their variety.
*	[Wikipedia](https://www.wikipedia.org/): with the help of [`wikipedia_query.py`](https://github.com/epfl-ada/ada-2022-project-nolemonnomelon/blob/main/wikipedia_query.py)we request imdb-associated films on wikipedia, with associated box office revenues and freebase IDs if available (on wikipedia). With the freebase IDs we will be able to associate this data with the CMU movie metadata.
*	[TMDB](https://developers.themoviedb.org/3/getting-started/introduction): [`TMDB_query.py`](https://github.com/epfl-ada/ada-2022-project-nolemonnomelon/blob/main/TMDB_query.py) uses the imdb ids and freebase ids to query movie budgets and country of origin production from the TMDB dataset.
* [Inflation data](https://data.worldbank.org/indicator/FP.CPI.TOTL.ZG): dataset containing inflation coefficient of each country from 1960 to 2021. This dataset is useful when we want to correct the movie box office revenue and the movie budget. Since the two are in US dollar we use the USA inflation as reference of all the movies. In addition, we converted all the values keeping as reference the year 2021.
All of this is taken into account by [`final_dataset_creation.py`](https://github.com/epfl-ada/ada-2022-project-nolemonnomelon/blob/main/final_dataset_creation.py).

All the details about the final dataset creation are available in the notebook [`dataset_creation.ipynb`](https://github.com/epfl-ada/ada-2022-project-nolemonnomelon/blob/main/dataset_creation.ipynb).

## Methods

**Step 1: Dataset construction and pre-processing.** [`dataset_creation.ipynb`](https://github.com/epfl-ada/ada-2022-project-nolemonnomelon/blob/main/dataset_creation.ipynb)

* CMU exploration.
* Final dataset creation.

**Step 2: Preliminary analysis and visualizations.** [`Exploratory.ipynb`](https://github.com/epfl-ada/ada-2022-project-nolemonnomelon/blob/main/Exploratory.ipynb)

* Graphical representations of revenue, budget and rating.
* Pearson correlation coefficient evaluation.

**Step 3: Genre analysis.**

* Graphical representations, correlation with revenue, budget and rating.
* Regression with selected features.  

**Step 4: PCA and clustering.**

* Find characteristics of low budget movies space.
* Determine regions with low budget and high revenue.

**Step 5: Data story.**

* Discussion.
* Conclusion.

**Step 6: Code optimization.**

## Proposed timeline

* 04.11.22 In deepth CMU data exploration.
* 11.11.22 Final data set creation and preliminary visualizations.
* 18.11.22 **Milestone 2 deadline.**
* 25.11.22 Genre analysis.
* 02.12.22 **Homework 2 deadline.**
* 09.12.22 PCA and clustering.
* 16.12.22 Finalisation, data story on web.
* 23.12.22 **Milestone 3 deadline.**

## Organization within the team

A list of internal milestones up until project Milestone 3.

<table class="tg" style="undefined;table-layout: fixed; width: 342px">
<colgroup>
<col style="width: 164px">
<col style="width: 178px">
</colgroup>
<thead>
  <tr>
    <th class="tg-0lax"></th>
    <th class="tg-0lax">Tasks</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0lax">@loconsta</td>
    <td class="tg-0lax">Develop the web interface<br><br>PCA analysis<br><br>Develop the final text for the data story</td>
  </tr>
  <tr>
    <td class="tg-0lax">@ghaeflig</td>
    <td class="tg-0lax">Develop the web interface<br><br>Clustering preliminary analysis<br><br>Develop the final text for the data story</td>
  </tr>
  <tr>
    <td class="tg-0lax">@rmonney</td>
    <td class="tg-0lax">Genre analysis<br><br>Include additional relevant CMU information<br><br>Develop the final text for the data story</td>
  </tr>
  <tr>
    <td class="tg-0lax">@LisaPaganiEPFL</td>
    <td class="tg-0lax">Synthesis of main result<br><br>Come up with meaningful visualizations<br><br>Develop the final text for the data story</td>
  </tr>
</tbody>
</table>
