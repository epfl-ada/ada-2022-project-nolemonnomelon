# Title TBD

___Study of the evolution of movie success across time___

## Abstract

We may think that a successful movie is defined by its revenue, the higher the income and the better the film. In this project, we want to go way beyond the mere revenue information. We are going to exploit the rating of movies as well as the inflation, to determine whether there is a temporal link between how much people seem to love a film and how much the film is actually earning. At this point we must not forget that the budget of a movie can impact it's success, it may be a link between the budget, the revenue and the rating. Once we have a better understanding of this mechanism, we would like to find which are the most successful movie genres in each of the 20th century decades. We may detect the golden age of the Far West and Science Fiction movies and it would be interesting to analyse how fast movie preferences change with time.

## Research Questions

Perform a temporal analysis of the changes of the popular movie genres.

1) Is there a link between revenue, budget and rating?
2) Which movie genres were likely to be watched and appreciated during the 20th century and the beginning of the 21st century?
3) How fast do popular movie genres change across time?

## Proposed additional datasets
The CMU movie metadata contains not many and not recent movies (until 2012 only). Moreover, it has a lot of NA values, espcially for the box office revenue. So, we decided to complete this dataset in order to have a more representative one.We use :

*	[IMBD](https://datasets.imdbws.com/): dataset to complete the amount of movies of the original [CMU](http://www.cs.cmu.edu/~ark/personas/) and ensure good representation of their variety.
*	[Wikipedia](https://www.wikipedia.org/): with the help of [`wikipedia_query.py`](https://github.com/epfl-ada/ada-2022-project-nolemonnomelon/blob/main/wikipedia_query.py)we request imdb-associated films on wikipedia, with associated box office revenues and freebase IDs if available (on wikipedia). With the freebase IDs we will be able to associate this data with the CMU movie metadata.
*	[TMDB](link?): [`TMDB_query.py`](https://github.com/epfl-ada/ada-2022-project-nolemonnomelon/blob/main/TMDB_query.py) uses the imdb ids and freebase ids to query movie budgets and country of origin production from the TMDB dataset.
* [Inflation data](https://data.worldbank.org/indicator/FP.CPI.TOTL.ZG): dataset containing inflation coefficient of each country from 1960 to 2021. This dataset is useful when we want to correct the movie box office revenue and the movie budget. Since the two are in US dollar we use the USA inflation as reference of all the movies. In addition, we converted all the values keeping as reference the year 2021. All of this is taking into account by [`final_dataset_creation.py`](https://github.com/epfl-ada/ada-2022-project-nolemonnomelon/blob/main/final_dataset_creation.py).

All the details about the final dataset creation are available in the notebook [`dataset_creation.ipynb`](https://github.com/epfl-ada/ada-2022-project-nolemonnomelon/blob/main/dataset_creation.ipynb).

## Methods

**Step 1: Dataset construction and pre-processing.**

* CMU exploration.
* Final dataset creation.

**Step 2: Preliminary analysis and visualizations.**

* Graphical representations of revenue, budget and rating.
* Pearson correlation coefficient evaluation.

**Step 3: Blabla.**

## Proposed timeline

* 04.11.22 In deepth CMU data exploration.
* 11.11.22 Final data set creation and preliminary visualizations.
* 18.11.22 **Homework 2 deadline**
* 25.11.22 .
* 02.12.22 .
* 09.12.22 .
* 16.12.22 .
* 23.12.22 **Milestone 3 deadline**

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
    <td class="tg-0lax">Come up with meaningful visualizations<br><br>Continue exploring the dataset<br><br>Develop the final text for the data story</td>
  </tr>
  <tr>
    <td class="tg-0lax">@ghaeflig</td>
    <td class="tg-0lax">Develop the web interface<br><br>Analyze news website bias<br><br>Develop the final text for the data story</td>
  </tr>
  <tr>
    <td class="tg-0lax">@rmonney</td>
    <td class="tg-0lax">Define topic of interests<br><br>Tune clustering<br><br>Develop the final text for the data story</td>
  </tr>
  <tr>
    <td class="tg-0lax">@LisaPaganiEPFL</td>
    <td class="tg-0lax">Develop the web interface<br><br>Integrate datasets of all years<br><br>Develop the final text for the data story</td>
  </tr>
</tbody>
</table>
