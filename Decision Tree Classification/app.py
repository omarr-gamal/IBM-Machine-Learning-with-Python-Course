import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import pandas as pd
import pylab as pl
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing, metrics, tree 

from  io import StringIO
import pydotplus


#read dataset
dataset = pd.read_csv("drug200.csv")


#feature set as numpy array
x = dataset[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values


#convert categorical data to numerical data
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
x[:,1] = le_sex.transform(x[:,1]) 

le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
x[:,2] = le_BP.transform(x[:,2])

le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
x[:,3] = le_Chol.transform(x[:,3]) 


#targets as numpy array
y = dataset["Drug"]


#split data into train data and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=3)


#create the entropy decision tree
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)


#train the decision tree
drugTree.fit(x_train,y_train)

predicted_data = drugTree.predict(x_test)


#evaluate the decision tree
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, predicted_data))


#visualize the decision tree
# dot_data = StringIO()
# filename = "drugtree.png"
# featureNames = dataset.columns[0:5]
# out = tree.export_graphviz(drugTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_train), filled=True,  special_characters=True,rotate=False)  
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
# graph.write_png(filename)
# img = mpimg.imread(filename)
# plt.figure(figsize=(100, 200))
# plt.imshow(img,interpolation='nearest')