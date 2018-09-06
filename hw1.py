
"""
Created on Thu Feb  1 20:59:41 2018

@author: hhanchan
"""
import numpy as np
from numpy import linalg as LA
import random
import time

from scipy.io import loadmat
ocr=loadmat('ocr.mat')

import matplotlib.pyplot as plt
from matplotlib import cm
plt.imshow(ocr['data'][0].reshape((28,28)), cmap=cm.gray_r)
plt.show()

data=ocr['data']
labels=ocr['labels']
testdata=ocr['testdata']
testlabels=ocr['testlabels']

def firststep(data,labels,n):
    sel = random.sample(range(60000),n)
    data=data[sel].astype('float')
    labels=labels[sel]
    return data,labels

def nnclassifier(train_data,test_data,train_labels,test_labels):
    Y=list(range(test_data.shape[0]))
    distance=list(range(train_data.shape[0]))
    for i in range(0,test_data.shape[0]):      
        mdist=np.inf #set the min distance to infinity, and it will be updated in every iteration
        for j in range(0,train_data.shape[0]):
            if np.dot(test_data[i]-train_data[j],test_data[i]-train_data[j])<mdist:
                Y[i]=train_labels[j][0] #update the yi
                mdist=np.dot(test_data[i]-train_data[j],test_data[i]-train_data[j])
            else:
                mdist=mdist
    error=np.sum(Y!=testlabels.T)/test_data.shape[0] #compute the error rate
    return error
   
def function(data, labels):
    array = [0] * 10
    for i in range(10): #iteration for 10 times
        array[i] = []
        print (i)
        for n in [1000, 2000, 4000, 8000]:
            train_data, train_labels = firststep(data, labels, n)
            error = nnclassifier(train_data, testdata, train_labels, testlabels)
            array[i].append(error)
    for i in range(10):  #plot the graph
        plt.plot([1000, 2000, 4000, 8000], array[i])

function(ocr['data'],ocr['labels'])
