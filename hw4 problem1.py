
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split # use it from sklearn to split data into training and validation set
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression 
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error # use it from sklearn to compute square loss risk
# read file
from scipy.io import loadmat
hw4data=loadmat('hw4data.mat')
X = hw4data['data']
y = hw4data['labels']

hw4data['quiz'].shape
quiz = hw4data['quiz']
n = hw4data['data'].shape[0]
int(0.75*n)
X_trainval = hw4data['data'][:196608]
y_trainval = hw4data['labels'][:196608]

X_train, X_valid, y_train, y_valid = train_test_split(X_trainval, y_trainval, random_state=0)

X_test = hw4data['data'][196608:]
y_test = hw4data['labels'][196608:]


# Linear Regression # use LinearRegression from sklearn
lr = LinearRegression()
lr.fit(X_train, y_train)
mean_squared_error(y_test,lr.predict(X_test))

# decision tree regressor # use DecisionTreeRegressor from sklearn
tree = DecisionTreeRegressor(max_depth=7)
tree.fit(X_train, y_train)
mean_squared_error(y_test,tree.predict(X_test))

# neural network # use MLPRegressor from sklearn
mlp = MLPRegressor(alpha=0.000001, random_state=0, solver="lbfgs")
mlp.fit(X_train, y_train)
mean_squared_error(y_test,mlp.predict(X_test))

# PQ
(np.sum(mlp.predict(quiz))+quiz.shape[0])/(2*quiz.shape[0])














