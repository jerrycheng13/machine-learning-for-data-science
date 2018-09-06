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

# part(b):
# compute the objective value
def f(b0, b, X, y):
    N = y.shape[0]
    f = 0
    for i in range(0, N):
        f += (1/N) * (np.log(1 + np.exp(b0 + np.dot(X[i],b))) - y[i]*(b0 + np.dot(X[i],b)))
    return f
#compute the gradient of beta_0
def b0_gradient(b0, b, X, y):
    N = y.shape[0]
    b0_gradient = 0
    for i in range(0, N):
        b0_gradient += (1/N) * ((np.exp(b0 + np.dot(X[i],b))/(1+np.exp(b0 + np.dot(X[i],b))))-y[i])
    return b0_gradient
# compute the gradient of beta
def b_gradient(b0, b, X, y):
    N = y.shape[0]
    b_gradient = 0
    for i in range(0, N):
        b_gradient += (1/N) * ((np.exp(b0 + np.dot(X[i],b))/(1+np.exp(b0 + np.dot(X[i],b))))*X[i]-y[i]*X[i])
    return b_gradient
# compute stepsize
def stepsize(b0, b, X, y):
    stepsize = 1
    while f(b0 - stepsize * b0_gradient(b0, b, X, y), b - stepsize * b_gradient(b0, b, X, y), X, y) > f(b0, b, X, y) - 1/2 * stepsize * (np.dot(b_gradient(b0, b, X, y), b_gradient(b0, b, X, y)) + np.dot(b0_gradient(b0, b, X, y), b0_gradient(b0, b, X, y))):
        stepsize = 1/2 * stepsize
    return stepsize
# gradient descent algorithm
def Gradientdescent(b0, b, X, y):
    iteration = 0 
    while f(b0, b, X, y) > 0.65064:
        sts = stepsize(b0, b, X, y)
        b0 -= sts * b0_gradient(b0, b, X, y)
        b -= sts * b_gradient(b0, b, X, y)
        iteration += 1
        print(iteration,f(b0, b, X, y))
    else:
        print("iterations: {}".format(iteration))
        print("final objective value: {}".format(f(b0, b, X, y)))
        print("updated beta_0 : {}".format(b0))
        print("updated beta: {}".format(b))
# use gradientdescent function to compute iterations    
Gradientdescent(0, [0,0,0], X, y)

# part(c)
# plot
data = pd.concat([pd.DataFrame(X), pd.DataFrame(y)], axis=1)
data.columns = ['feature1', 'feature2', 'feature3', 'label']
dat1 = data[data.label==0]
dat2 = data[data.label==1]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(dat1['feature1'], dat1['feature2'], dat1['feature3'], c='r', label='0')
ax.scatter(dat2['feature1'], dat2['feature2'], dat2['feature3'], c='b', label='1')
ax.set_xlabel('feature_1')
ax.set_ylabel('feature_2')
ax.set_zlabel('feature_3')
plt.legend(('0', '1'), loc='best')
plt.show()
# statistic summary
data.describe()
# The result of the plot and statistic summary shows that we could scale the data to increase the step size, in order to use less iteration
# define linear transformation
A = [[1/20, 0, 0],[0, 1, 0],[0, 0, 1/20]]
aX = []
for i in range(0, X.shape[0]):
    aX.append(np.dot(A, X[i]))
aX = np.array(aX)
Gradientdescent(0, [0,0,0], aX, y)

# part(d)
n = y.shape[0]
0.8*n
# first[0.8n] as training data
X_train = X[:3277]
y_train = y[:3277]
aX_train = aX[:3277]
# remaining n-[0.8n] as testing data
X_test = X[3277:]
y_test = y[3277:]
aX_test = aX[3277:]
# validation error rate
def error(b0, b, X_test, y_test):
    y0 = []
    for i in range(0, y_test.shape[0]):
        if 1/(1+np.exp(-b0-np.dot(X_test[i], b))) > 1/2:
            y0.append(1)
        else:
            y0.append(0)
    return np.mean(np.array(y0) != y_test)
# report with summary
def Gradientdescent_summary(b0, b, X_train, y_train, X_test, y_test):
    iteration = 0 
    rate = 0
    err = []
    err.append(error(b0, b, X_test, y_test))
    while rate <= 0.99 or iteration <=32:
        sts = stepsize(b0, b, X_train, y_train)
        b0 -= sts * b0_gradient(b0, b, X_train, y_train)
        b -= sts * b_gradient(b0, b, X_train, y_train)
        iteration += 1
        if np.log2(iteration) % 1 == 0:
            rate = error(b0, b, X_test, y_test)/min(err)
            err.append(error(b0, b, X_test, y_test))
            print("iterations: {}".format(iteration), "objective value: {}".format(f(b0, b, X_train, y_train)), "validation error rate: {}".format(error(b0, b, X_test, y_test)))
    else:
        print("iterations: {}".format(iteration))
        print("final objective value: {}".format(f(b0, b, X_train, y_train)))
        print("final validation error rate: {}".format(error(b0, b, X_test, y_test)))
# running on original data
Gradientdescent_summary(0, [0,0,0], X_train, y_train, X_test, y_test)
# running on linear transformed data
Gradientdescent_summary(0, [0,0,0], aX_train, y_train, aX_test, y_test)  

   

















        
        