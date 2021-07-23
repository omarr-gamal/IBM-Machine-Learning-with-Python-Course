import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import pandas as pd
import pylab as pl
import numpy as np

import scipy.optimize as opt

from sklearn.datasets._samples_generator import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import preprocessing, metrics

from io import StringIO
import pydotplus

import itertools
import random


# create random dataset
np.random.seed(0)

# x is feature matrix of shape [n_samples, n_features]
# y is the response vector of shape [n_samples]
x, y = make_blobs(n_samples=5000,
                  centers=[[4, 4], [-2, -1], [2, -3], [1, 1]],
                  cluster_std=0.9)

# display the scatter plot of the randomly generated data
plt.scatter(x[:, 0], x[:, 1], marker='.')
plt.show()


# setup k-means clustering
k_means = KMeans(init="k-means++", n_clusters=4, n_init=12)


# fit the model with the feature matrix x
k_means.fit(x)

k_means_labels = k_means.labels_
k_means_cluster_centers = k_means.cluster_centers_


# create the visual plot
# Initialize the plot with the specified dimensions.
fig = plt.figure(figsize=(6, 4))

# Colors uses a color map, which will produce an array of colors based on
# the number of labels there are. We use set(k_means_labels) to get the
# unique labels.
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

# Create a plot
ax = fig.add_subplot(1, 1, 1)

# For loop that plots the data points and centroids.
# k will range from 0-3, which will match the possible clusters that each
# data point is in.
for k, col in zip(range(len([[4, 4], [-2, -1], [2, -3], [1, 1]])), colors):

    # Create a list of all data points, where the data points that are
    # in the cluster (ex. cluster 0) are labeled as true, else they are
    # labeled as false.
    my_members = (k_means_labels == k)

    # Define the centroid, or cluster center.
    cluster_center = k_means_cluster_centers[k]

    # Plots the datapoints with color col.
    ax.plot(x[my_members, 0], x[my_members, 1],
            'w', markerfacecolor=col, marker='.')

    # Plots the centroids with specified color, but with a darker outline
    ax.plot(cluster_center[0], cluster_center[1], 'o',
            markerfacecolor=col,  markeredgecolor='k', markersize=6)

# Title of the plot
ax.set_title('KMeans')

# Remove x-axis ticks
ax.set_xticks(())

# Remove y-axis ticks
ax.set_yticks(())

# Show the plot
plt.show()
