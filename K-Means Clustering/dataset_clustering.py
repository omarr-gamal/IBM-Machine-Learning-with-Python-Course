from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import pandas as pd
import pylab as pl
import numpy as np

import scipy.optimize as opt
from scipy.sparse import data

from sklearn.datasets._samples_generator import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import preprocessing, metrics

from io import StringIO
import pydotplus

import itertools
import random


# read the dataset
dataset = pd.read_csv("Cust_Segmentation.csv")

# drop the address feature from the dataset because it's categorical
dataset = dataset.drop('Address', axis=1)

# normalize the data over the standard deviation
x = dataset.values[:, 1:]
x = np.nan_to_num(x)
Clus_dataSet = preprocessing.StandardScaler().fit_transform(x)

# fit the data to the k-means model
clusterNum = 3
k_means = KMeans(init="k-means++", n_clusters=clusterNum, n_init=12)
k_means.fit(x)
labels = k_means.labels_

# assign labels to each row in the dataframe
dataset["Clus_km"] = labels

dataset.groupby('Clus_km').mean()

# distribute customers based on age and income
area = np.pi * (x[:, 1])**2
plt.scatter(x[:, 0], x[:, 3], s=area, c=labels.astype(np.float), alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)

plt.show()
