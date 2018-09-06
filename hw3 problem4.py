import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd


from scipy.io import loadmat
# read data
dat=loadmat('hw3data.mat')
# get the shape
dat['data'].shape
X = dat['data']
y = dat['labels']

# if labeled 0, change it to -1
y0=list(range(y.shape[0]))
for i in range(0, y.shape[0]):
    if y[i]==1:
        y0[i]=1
    else:
        y0[i]=-1
y0 = np.array(y0)
# compute objective value
def f(a, X, y):
    N = y.shape[0]
    f = 0
    for i in range(0, N):
        for j in range(0, N):
            f -= y[i]*y[j]*np.dot(X[i],X[j])*a[i]*a[j]
        f += a[i]
    return f
# compute weight vector
def weight(a,X,y):
    N = y.shape[0]
    w = 0
    for i in range(0, N):
        w += a[i]*y[i]*X[i]
    return w
# dual coordinate ascent algorithm
def dualcoordinateascent(X,y,iteration_t):
    N = y.shape[0]
    C = 10/N
    a = [0] * N
    iteration = 0
    while iteration < iteration_t:
        for i in range(0, N):
            df = 1
            for j in range(0, N):
                if i != j:
                    df -= y[i]*y[j]*np.dot(X[i],X[j])*a[j]
                else:
                    v = 1/(2*y[i]*y[i]*np.dot(X[i],X[i]))
            if df > 0 or v >= C:
                a[i] = C
            if v > 0 and v < C:
                a[i] = v
            if df <= 0 or v < 0:
                a[i] = 0
            if i/40 % 1 == 0:
                print("iteration {0} already: {1}".format(iteration+1, i/N))
        iteration += 1
    else:
        print("iterations: {}".format(iteration))
        print("objective value: {}".format(f(a, X, y)))
        print("weight vector: {}".format(weight(a, X, y)))
        
# running on original data while change labeled 0 to -1
dualcoordinateascent(X,y0,2)      
# standardization feature transformation
X0 = pd.DataFrame(X)
X0.columns = ['feature1', 'feature2', 'feature3']
X0['feature1']=(X0['feature1']-np.mean(X0['feature1']))/np.std(X0['feature1'])
X0['feature2']=(X0['feature2']-np.mean(X0['feature2']))/np.std(X0['feature2'])
X0['feature3']=(X0['feature3']-np.mean(X0['feature3']))/np.std(X0['feature3'])
X0 = np.array(X0)
# running on data where features are standardized
dualcoordinateascent(X0,y0,2)            
            
    
    
    
    
    
    
    
    
    
    
    
    
