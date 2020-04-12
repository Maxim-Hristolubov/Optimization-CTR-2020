#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import sklearn.datasets as skldata
import sklearn.preprocessing as skprep
import scipy.optimize as scopt
import sklearn.preprocessing as skprep
import scipy.special as scspec
import cvxpy as cvx
from scipy.sparse import *
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import argparse
import random

def create_data(m,n):
    (A_dense, y1) = make_blobs(n_samples=n, n_features=m, centers=2, cluster_std=20, random_state=95)
    
    (A_sprase, y2) = make_blobs(n_samples=n, n_features=n, centers=2, cluster_std=20, random_state=95)
    A_sprase[:][np.random.rand(n) < (n-m)/n] = 0
    
    A = np.concatenate((A_dense,A_sprase), axis = 1)
    A = skprep.normalize(A, norm="l2", axis=0)
    y1 = np.asarray([y if y else -1 for y in y1])
    y2 = np.asarray([y if y else -1 for y in y2])
    x_true = y1*y2
    X, y = csr_matrix(A), x_true
    print(f'X.shape = {X.shape}',f'y.shape = {y.shape}', f'nonzero in X = {X.nonzero()[0].shape}')
    return X*50,y

def prox_alg(f, f_grad, g_prox, x0, C, num_iter, alpha=1, accel=False):
    x = x0.copy()
    conv, funct = [], []
    conv.append(x)
    funct.append(x)
    if accel:
        t_prev = 1
        t_next = (1 + np.sqrt(1 + 4 * t_prev**2)) / 2.
    for i in range(num_iter):
        if accel and i > 0:
            x = x + (t_prev - 1) / t_next * (x - conv[-2])
            t_prev = t_next
            t_next = (1 + np.sqrt(1 + 4 * t_prev**2)) / 2.
   
        z = g_prox(x - alpha * f_grad(x), C)
        x = z.copy()
        conv.append(x)
    return x, conv


import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import argparse

def next_batch(X, y, i, batchSize):
    if i + batchSize >= X.shape[0]:
        i = 0
    return (X[i:i + batchSize], y[i:i + batchSize], i + batchSize)

def SGD(f, f_gradient, X, y, C, x0, epochs, alpha, batch_size):
    w = x0.copy()
    conv, funct = [], []
    conv.append(w)
    for epoch in np.arange(0, epochs):
        j = 0
        while(j + batch_size < X.shape[0]):
            (batchX, batchY, j) = next_batch(X, y, j, batch_size)
            gradient = f_gradient(w, batchX, batchY, C)
            z = w - alpha * gradient
            w = z.copy()
            conv.append(w)
    return w, conv

def prox_SGD(f, f_gradient, g_prox, X, y, C, x0, epochs, alpha, batch_size, accel=False):
    w = x0.copy()
    conv, funct = [], []
    conv.append(w)
    if accel:
        t_prev = 1
        t_next = (1 + np.sqrt(1 + 4 * t_prev**2)) / 2.
    for i, epoch in enumerate(np.arange(0, epochs)):
        j = 0
        while(j + batch_size < X.shape[0]):
            (batchX, batchY, j) = next_batch(X, y, j, batch_size)
            if accel and i > 0:
                w = w + (t_prev - 1) / t_next * (w - conv[-2])
                t_prev = t_next
                t_next = (1 + np.sqrt(1 + 4 * t_prev**2)) / 2.
            gradient = f_gradient(w, batchX, batchY, C)
            z = g_prox(w - alpha * gradient, C)
            w = z.copy()
            conv.append(w)
    return w, conv

from numpy import linalg as LA

def adapt_SGD(f, f_gradient, g_prox, X, y, C, x0, epochs, L_0 = 0.0001, eps = 0.1, D_0 = 0.001, accel=False):
    w_pred, L_pred = x0.copy(), L_0
    conv = []
    w = w_pred.copy()
    conv.append(w)
    j = 0
    for epoch in range(epochs):
        L_next = L_pred/4
        while True:
            L_next = L_next*2
            r = int(max(D_0//(L_next*eps),1))
            (batchX, batchY, j) = next_batch(X, y, j, r)
            gradient = f_gradient(w_pred, batchX, batchY, C)
            w_next = w_pred - 1/(8*L_next)*gradient
            if f(w_next,batchX,batchY,C)<=f(w_pred,batchX,batchY,C)+gradient.dot(w_next-w_pred)+L_next*LA.norm(w_next-w_pred, 2)**2+eps/2:
                break
        w = w_next.copy()
        conv.append(w)
        w_pred, L_pred = w_next, L_next
    return conv[-1], conv


def adapt_SGDac(f, f_gradient, g_prox, X, y, C, x0, epochs, L_0 = 0.0001, eps = 0.1, D_0 = 0.01, accel=True):
    w_pred, L_pred = x0.copy(), L_0
    conv = []
    w = w_pred.copy()
    conv.append(w)
    A_pred,y_pred,u_pred = 0, x0.copy(), x0.copy()
    j = 0
    for epoch in range(epochs):
        L_next = L_pred/4.
        while True:
            L_next = L_next*2
            a = (1+np.sqrt(1+4*A_pred*L_next))/2/L_next
            A_next = A_pred + a
            r = int(max(D_0*a//eps,1))
            y = (a*u_pred+A_pred*w_pred)/A_next
            
            (batchX, batchY, j) = next_batch(X, y, j, r)
            gradient = f_gradient(y, batchX, batchY, C)
            u_next = u_pred - 0.5*a*gradient
            w_next = (a*u_next+A_pred*w_pred)/A_next
            if f(w_next,batchX,batchY,C)<=f(y,batchX,batchY,C)+gradient.dot(w_next-y)+L_next*LA.norm(w_next-y, 2)**2+a/(2*A_next)*eps:
                break 
        w = w_next.copy()
        conv.append(w)
        w_pred, L_pred, A_pred, u_pred = w_next, L_next, A_next, u_next,
    return conv[-1], conv


