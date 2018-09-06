import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import zero_one_loss
from sklearn.preprocessing import StandardScaler

from scipy.io import loadmat
hw5data=loadmat('hw5data.mat')

X1 = hw5data['Sdata']
y1 = hw5data['Slabels']

X2 = hw5data['Tdata']
y2 = hw5data['Tlabels']

X_test = hw5data['testdata']
y_test = hw5data['testlabels']

scaler = StandardScaler()

Xs=pd.DataFrame(np.concatenate((X1,y1),axis=1))
Xt=pd.DataFrame(np.concatenate((X2,y2),axis=1))

Xs = Xs[Xs[784]==1]
Xt = Xt[Xt[784]==-1]
X = np.concatenate((Xs, Xt), axis=0)
X = pd.DataFrame(X)
X_train = X[X.columns[0:784]]
y_train = X[X.columns[784]]

#X_train = np.concatenate((X1, X2), axis=0)
#y_train = np.concatenate((y1, y2), axis=0)
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

lgr = LogisticRegression(C=0.005, class_weight={-1:0.85,1:0.15},n_jobs=-1)
lgr.fit(X_train_scaled,y_train)
lgr.score(X_train_scaled,y_train)
lgr.score(X_test_scaled,y_test.ravel())

param_grid = {'C' : [0.01,0.1,1,10,100]}
grid = GridSearchCV(LogisticRegression(class_weight={-1:0.15,1:0.85},n_jobs=-1), param_grid, cv=5, return_train_score=True)
grid.fit(X_train_scaled,y_train.ravel())
print(grid.best_params_)
grid.score(X_test_scaled,y_test.ravel())


def l3_score(X,y,weight1,weight2):
#    c = weight1*l1.predict_proba(X) +weight2*l2.predict_proba(X)
#    l3 = []
#    for i in range(0,y.shape[0]):
#        if c[i][1] >= c[i][0]:
#            l3.append(1)
#        else:
#            l3.append(-1)
#    l3 = np.array(l3)
    
    l3 = weight1 * l1.decision_function(X)+weight2 * l2.decision_function(X)    
    np.place(l3, l3 > 0, 1)
    np.place(l3, l3 <= 0, -1)
    return accuracy_score(y.ravel(), l3)

l3_score(X_test,y_test,0.85,0.15)

#l4 = RandomForestClassifier(n_estimators=50, random_state=0)
#l4.fit(X1,y1)

#l5 = RandomForestClassifier(n_estimators=50, random_state=0)
#l5.fit(X2,y2)

a = l1.predict_proba(X_test)
b = l2.predict_proba(X_test)
c = 0.85*a +0.15*b
l3 = []
for i in range(0,y_test.shape[0]):
    if c[i][1] >= c[i][0]:
        l3.append(1)
    else:
        l3.append(-1)
l3 = np.array(l3)

accuracy_score(y_test.ravel(), l3)
    









