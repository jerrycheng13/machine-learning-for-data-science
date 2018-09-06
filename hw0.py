#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 20:55:22 2018

@author: hhanchan
"""

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# (c)
A=np.matrix([[-1,0,0,0,0,0],[0,0.5,0,0,0,0],[0,0,0.5,0,0,0],[0,0,0,0.125,0.125,0],[0,0,0,0.125,0.125,0],[0,0,0,0,0,0.25]])
eta1=0.5
b=np.matrix([[1,1,1,1,1,1]]).transpose()
g=list(range(1001))
x=list(range(1001))
x[0]=np.matrix([[0,0,0,0,0,0]]).transpose()
g[0]=(LA.norm(A*x[0]-b))*(LA.norm(A*x[0]-b))
for k in range(1,1001):
    x[k]=x[k-1]+eta1*A.transpose()*(b-A*x[k-1])
    g[k]=(LA.norm(A*x[k]-b))*(LA.norm(A*x[k]-b))
    
v=np.matrix([[0,0,0,1,-1,0]]).transpose()
xhat=x[1000]+v
print (g[1000])
ghat=(LA.norm(A*xhat-b))*(LA.norm(A*xhat-b))
print (ghat)

print (LA.norm(xhat)*LA.norm(xhat))
print (LA.norm(x[1000])*LA.norm(x[1000]))

# (d)
eta2=0.75
y=list(range(101))
y[0]=np.matrix([[0,0,0,0,0,0]]).transpose()
for k in range(1,100):
    y[k]=y[k-1]+eta2*A.transpose()*(b-A*y[k-1])
    
kindex=np.arange(1,101,1)
f1=g[1:101]
f2=list(range(100))
f2[0]=(LA.norm(A*y[0]-b))*(LA.norm(A*y[0]-b))
for k in range(1,100):
    f2[k]=(LA.norm(A*y[k]-b))*(LA.norm(A*y[k]-b))
plt.plot(kindex,f1,'g-.',kindex,f2,'ro')
green=mpatches.Patch(color='green',label='$||Ax^{(k)}-b||_2^2$')
red=mpatches.Patch(color='red',label='$||Ay^{(k)}-b||_2^2$')
plt.legend(handles=[green,red])
plt.xlabel('k')
plt.ylabel('g(x)')
plt.title('g(x): Richardson Iteration')
plt.show()