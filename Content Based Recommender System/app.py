import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import pandas as pd
import pylab as pl
import numpy as np

from scipy import ndimage
from scipy.cluster import hierarchy
from scipy.spatial import distance_matrix

from sklearn import manifold, datasets, preprocessing, metrics
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets._samples_generator import make_blobs

from io import StringIO
from math import sqrt
import pydotplus

import itertools


# Storing the movie information into a pandas dataframe
movies_df = pd.read_csv('movies.csv')
# Storing the user information into a pandas dataframe
ratings_df = pd.read_csv('ratings.csv')

# Using regular expressions to find a year stored between parentheses
# We specify the parantheses so we don't conflict with movies that have years in their titles
movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))', expand=False)
# Removing the parentheses
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)', expand=False)
# Removing the years from the 'title' column
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')
# Applying the strip function to get rid of any ending whitespace characters that may have appeared
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())


# Every genre is separated by a | so we simply have to call the split function on |
movies_df['genres'] = movies_df.genres.str.split('|')


# Copying the movie dataframe into a new one since we won't need to use the genre information in our first case.
moviesWithGenres_df = movies_df.copy()

# For every row in the dataframe, iterate through the list of genres and place a 1 into the corresponding column
for index, row in movies_df.iterrows():
    for genre in row['genres']:
        moviesWithGenres_df.at[index, genre] = 1
# Filling in the NaN values with 0 to show that a movie doesn't have that column's genre
moviesWithGenres_df = moviesWithGenres_df.fillna(0)


# Drop removes a specified row or column from a dataframe
ratings_df = ratings_df.drop('timestamp', 1)


userInput = [
    {'title': 'Breakfast Club, The', 'rating': 5},
    {'title': 'Toy Story', 'rating': 3.5},
    {'title': 'Jumanji', 'rating': 2},
    {'title': "Pulp Fiction", 'rating': 5},
    {'title': 'Akira', 'rating': 4.5}
]
inputMovies = pd.DataFrame(userInput)


# Filtering out the movies by title
inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
# Then merging it so we can get the movieId. It's implicitly merging it by title.
inputMovies = pd.merge(inputId, inputMovies)
# Dropping information we won't use from the input dataframe
inputMovies = inputMovies.drop('genres', 1).drop('year', 1)


# Filtering out the movies from the input
userMovies = moviesWithGenres_df[moviesWithGenres_df['movieId'].isin(
    inputMovies['movieId'].tolist())]


# Resetting the index to avoid future issues
userMovies = userMovies.reset_index(drop=True)
# Dropping unnecessary columns to save memory
userGenreTable = userMovies.drop('movieId', 1).drop(
    'title', 1).drop('genres', 1).drop('year', 1)


# Dot produt to get weights
userProfile = userGenreTable.transpose().dot(inputMovies['rating'])


# Now let's get the genres of every movie in our original dataframe
genreTable = moviesWithGenres_df.set_index(moviesWithGenres_df['movieId'])
# And drop the unnecessary information
genreTable = genreTable.drop('movieId', 1).drop(
    'title', 1).drop('genres', 1).drop('year', 1)


# Multiply the genres by the weights and then take the weighted average
recommendationTable_df = (
    (genreTable*userProfile).sum(axis=1))/(userProfile.sum())

# Sort our recommendations in descending order
recommendationTable_df = recommendationTable_df.sort_values(ascending=False)


# The final recommendation table
movies_df.loc[movies_df['movieId'].isin(
    recommendationTable_df.head(20).keys())]
