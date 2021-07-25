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
import pydotplus

import itertools


# generate the feature matrix x1 and the responce vector y1
X1, y1 = make_blobs(n_samples=50,
                    centers=[[4, 4], [-2, -1], [1, 1], [10, 4]],
                    cluster_std=0.9)

# plot the scatter plot of the randomly generated data
plt.scatter(X1[:, 0], X1[:, 1], marker='o')

# create the clustering model
agglom = AgglomerativeClustering(n_clusters=4, linkage='average')

# fit the data to the model
agglom.fit(X1, y1)


# Create a figure of size 6 inches by 4 inches.
plt.figure(figsize=(6, 4))

# These two lines of code are used to scale the data points down,
# Or else the data points will be scattered very far apart.

# Create a minimum and maximum range of X1.
x_min, x_max = np.min(X1, axis=0), np.max(X1, axis=0)

# Get the average distance for X1.
X1 = (X1 - x_min) / (x_max - x_min)

# This loop displays all of the datapoints.
for i in range(X1.shape[0]):
    # Replace the data points with their respective cluster value
    # (ex. 0) and is color coded with a colormap (plt.cm.spectral)
    plt.text(X1[i, 0], X1[i, 1], str(y1[i]),
             color=plt.cm.nipy_spectral(agglom.labels_[i] / 10.),
             fontdict={'weight': 'bold', 'size': 9})

# Remove the x ticks, y ticks, x and y axis
plt.xticks([])
plt.yticks([])
# plt.axis('off')


# Display the plot of the original data before clustering
plt.scatter(X1[:, 0], X1[:, 1], marker='.')
# Display the plot
plt.show()


dist_matrix = distance_matrix(X1, X1)
Z = hierarchy.linkage(dist_matrix, 'complete')

dendro = hierarchy.dendrogram(Z)
