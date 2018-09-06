import pandas as pd
import numpy as np

from scipy.io import loadmat
hw6data=loadmat('hw6data.mat')

H1 = hw6data['seq1_h']
X1 = hw6data['seq1_x']

# part(b)

## A
def A(a,b):
    A = 0
    B = 0
    for t in range(0, H1.shape[0]-1):
        if H1[t] == a and H1[t+1] == b:
            A += 1
        if H1[t] == a:
            B += 1
    return A/B
### A11
A11 = A(1,1)
### A12
A12 = A(1,2)
### A21
A21 = A(2,1)
### A22
A22 = A(2,2)

## mu
def mu(a):
    num = 0
    denom = 0
    for t in range(0, H1.shape[0]):
        if H1[t] == a:
            num += X1[t]
            denom += 1
    return num/denom
## mu_1
mu_1 = mu(1)
## mu_2
mu_2 = mu(2)

print("MLE for A11: {:}".format(A11))
print("MLE for A12: {:}".format(A12))
print("MLE for A21: {:}".format(A21))
print("MLE for A22: {:}".format(A22))
print("MLE for mu1: {:}".format(mu_1))
print("MLE for mu2: {:}".format(mu_2))


# part(c)
H2 = hw6data['seq2_h']
X2 = hw6data['seq2_x']

def prob_x_h(X, t, h):
    if h == 1:
        return (1/(2*np.pi)**4)*np.exp(-(1/2)*np.dot((X[t]-mu_1),(X[t]-mu_1)))
    if h == 2:
        return (1/(2*np.pi)**4)*np.exp(-(1/2)*np.dot((X[t]-mu_2),(X[t]-mu_2)))
    
## m_x_h as Observation distributions
m_x_h = []
for t in range(0, H2.shape[0]):
    m_x_h.append([prob_x_h(X2, t, 1), prob_x_h(X2, t, 2)])
m_x_h = pd.DataFrame(m_x_h)

## f_m_h as Forward messages
def forward(X):
    m_h = []
    m_h.append([A21, A22])
    for t in range(1, X.shape[0]):
        m_h_1 = A11*m_h[t-1][0]*prob_x_h(X, t-1, 1)+A21*m_h[t-1][1]*prob_x_h(X, t-1, 2)
        m_h_2 = A12*m_h[t-1][0]*prob_x_h(X, t-1, 1)+A22*m_h[t-1][1]*prob_x_h(X, t-1, 2)
        m_h.append([m_h_1/(m_h_1+m_h_2), m_h_2/(m_h_1+m_h_2)])
    m_h = pd.DataFrame(m_h)
    return m_h

f_m_h = forward(X2)

## b_m_h as Backward messages
def backward(X):
    m_h = []
    m_h.append([1, 1])
    X = X[::-1]
    for t in range(1, X.shape[0]):
        m_h_1 = A11*m_h[t-1][0]*prob_x_h(X, t-1, 1)+A21*m_h[t-1][1]*prob_x_h(X, t-1, 2)
        m_h_2 = A12*m_h[t-1][0]*prob_x_h(X, t-1, 1)+A22*m_h[t-1][1]*prob_x_h(X, t-1, 2)
        m_h.append([m_h_1/(m_h_1+m_h_2), m_h_2/(m_h_1+m_h_2)])
    m_h = pd.DataFrame(m_h)
    m_h = m_h.iloc[::-1]
    m_h = m_h.reset_index(drop=True)
    return m_h

b_m_h = backward(X2)

## p_h
### using only f_m_h and m_x_h
H2 = pd.DataFrame(H2)
p_h_1 = []
for t in range(0,X2.shape[0]):
    if (f_m_h[0][t] * m_x_h[0][t]) > (f_m_h[1][t] * m_x_h[1][t]):
        p_h_1.append(1)
    else:
        p_h_1.append(2)
p_h_1 = pd.DataFrame(p_h_1)

        
print("Part(c): Total number of prediction mistakes using only Forward messages and observation distributions: {:}".format(np.sum(p_h_1 != H2)[0]))

### using f_m_h, b_m_h, m_x_h, if we could treat t as T
p_h_2 = []
for t in range(0,X2.shape[0]):
    if (f_m_h[0][t] * b_m_h[0][t] * m_x_h[0][t]) > (f_m_h[1][t] * b_m_h[1][t] * m_x_h[1][t]):
        p_h_2.append(1)
    else:
        p_h_2.append(2)
p_h_2 = pd.DataFrame(p_h_2)
print("Part(c): Total number of prediction mistakes using both Forward and Backward messages and observation distributions: {:}".format(np.sum(p_h_2 != H2)[0]))


# part(d)
## pi
def pi(a):
    pi = 0
    for i in range(0, H1.shape[0]):
        if H1[i][0]==a:
            pi += 1
    return pi/H1.shape[0]

pi_1 = pi(1)
pi_2 = pi(2)
pi = (pi_1,pi_2)
print("The class priors are: {:}".format(pi))

## predict
H2_pred = []
for t in range(0,X2.shape[0]):
    p_1_x = pi_1*prob_x_h(X2, t, 1)
    p_2_x = pi_2*prob_x_h(X2, t, 2)
    if p_1_x >= p_2_x:
        H2_pred.append(1)
    else:
        H2_pred.append(2)
H2_pred = pd.DataFrame(H2_pred)       

print("Part(d): Total number of prediction mistakes: {:}".format(np.sum(H2_pred != H2)[0]))









