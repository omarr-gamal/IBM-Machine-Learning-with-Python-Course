import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import pandas as pd
import pylab as pl
import numpy as np

import scipy.optimize as opt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing, metrics 

from  io import StringIO
import pydotplus

import itertools


#read dataset
dataset = pd.read_csv("ChurnData.csv")

#select the 10 important features
dataset = dataset[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'callcard', 'wireless', 'churn']]

#convert churn values to int
dataset['churn'] = dataset['churn'].astype('int')


#feature set as numpy array
x = np.asarray(dataset[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])

#targets as numpy array
y = np.asarray(dataset['churn'])


#normalize the feature set
x = preprocessing.StandardScaler().fit(x).transform(x)


#split data into train data and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)


#create the logistic regression model
model = LogisticRegression(C=0.01, solver='liblinear').fit(x_train,y_train)


#predict class of y
yhat = model.predict(x_test)

#predict the probability of each class for y
yhat_prob = model.predict_proba(x_test)


#calculate jaccard index for yhat
j_score = metrics.jaccard_score(y_test, yhat, pos_label=0)



#prints the model's confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


#Compute confusion matrix
cnf_matrix = metrics.confusion_matrix(y_test, yhat, labels=[1,0])
np.set_printoptions(precision=2)


#Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')


print (metrics.classification_report(y_test, yhat))

#the model's log loss
l_los = metrics.log_loss(y_test, yhat_prob)





