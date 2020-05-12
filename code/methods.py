#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import timeit

def time_manager(function_to_decorate):
    def the_wrapper_around_the_original_function(*args, **kwargs):
        time = timeit.default_timer()
        ret = function_to_decorate(*args, **kwargs) # Сама функция
        res_t = timeit.default_timer()-time
        return res_t, ret
    return the_wrapper_around_the_original_function

C = (0.001,0.00001)
m = 64
n = 1024

import numpy as np
import sklearn.datasets as skldata
import sklearn.preprocessing as skprep
import scipy.optimize as scopt
import sklearn.preprocessing as skprep
import scipy.special as scspec
from scipy.sparse import *
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
from scipy.sparse import coo_matrix, hstack
import argparse
import random

def f(w, Xb, yb):
    return np.mean(np.logaddexp(np.zeros(Xb.shape[0]), -yb * Xb.dot(w)))
def g(w, C=C, m = 64):
    return C[0]*np.linalg.norm(w[:m],2)**2+C[1]*np.linalg.norm(w[m:],2)**2
def logloss(w, Xb, yb, C=C):
    return f(w, Xb, yb) + g(w, C)
def f_gradient(w, Xb, yb):
    denom = scspec.expit(-yb * Xb.dot(w))
    return -  Xb.T.dot(yb * denom) / Xb.shape[0]
def g_gradient(w,C=C, m = 64):
    return np.concatenate((2*C[0]*w[:m],2*C[1]*w[m:]), axis=0)
def logloss_gradient(w, Xb, yb, C=C):
    return g_gradient(w,C) + f_gradient(w, Xb, yb)
def quad_prox(w, C=C, m = 64):
    return np.concatenate((w[:m]/(2*C[0]+1),w[m:]/(2*C[1]+1)), axis=0)

def create_data(m,n, filepath = None):
    if filepath != None:
        import scipy.io as sio
        mat_contents = sio.loadmat(filepath)
        A_sprase = mat_contents['Problem'][0][0][2]
        y2 = np.asarray([1 if el > 1 else -1 for el in np.asarray(A_sprase.sum(axis = 1).T)[0]])
        n = A_sprase.shape[0]
    else:
        (A_sprase, y2) = make_blobs(n_samples=n, n_features=n//16, centers=2, cluster_std=20, random_state=95)
        A_sprase[:][np.random.rand(n) < (n-m)/n] = 0
        y2 = np.asarray([y if y else -1 for y in y2])
            
    (A_dense, y1) = make_blobs(n_samples=n, n_features=m, centers=2, cluster_std=20, random_state=95)
    y1 = np.asarray([y if y else -1 for y in y1])
    A = hstack([A_dense,A_sprase])
    A = skprep.normalize(A, norm="l2", axis=0)
    y_true = y1*y2
    X, y = csr_matrix(A), y_true
    print(f'X.shape = {X.shape}',f'y.shape = {y.shape}', f'nonzero in X = {X.nonzero()[0].shape}')
    return X, y

def prox_alg(f, f_grad, g_prox, x0, C, num_iter, alpha=1, accel=False):
    time = timeit.default_timer()
    x = x0.copy()
    conv, t_conv, funct = [], [], []
    conv.append(x)
    t_conv.append(timeit.default_timer()-time)
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
        t_conv.append(timeit.default_timer()-time)
    return t_conv, x, conv


import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import argparse

def next_batch(X, y, i, batchSize):
    if i + batchSize >= X.shape[0]:
        if batchSize >= X.shape[0]:
            return (X, y, 0)
        i = X.shape[0] - batchSize
    return (X[i:i + batchSize], y[i:i + batchSize], i + batchSize)

def SGD(f, f_gradient, X, y, C, x0, epochs, alpha, batch_size):
    time = timeit.default_timer()
    w = x0.copy()
    conv, funct, t_conv = [], [], []
    conv.append(w)
    t_conv.append(timeit.default_timer()-time)
    for epoch in np.arange(0, epochs):
        j = 0
        while(j + batch_size < X.shape[0]):
            (batchX, batchY, j) = next_batch(X, y, j, batch_size)
            gradient = f_gradient(w, batchX, batchY, C)
            z = w - alpha * gradient
            w = z.copy()
            conv.append(w)
            t_conv.append(timeit.default_timer()-time)
    return t_conv, w, conv

def prox_SGD(f, f_gradient, g_prox, X, y, C, x0, epochs, alpha, batch_size, accel=False): #f not full
    time = timeit.default_timer()
    w = x0.copy()
    conv, funct, t_conv = [], [], []
    conv.append(w)
    t_conv.append(timeit.default_timer()-time)
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
            gradient = f_gradient(w, batchX, batchY)
            z = g_prox(w - alpha * gradient, C)
            w = z.copy()
            conv.append(w)
            t_conv.append(timeit.default_timer()-time)
    return t_conv, w, conv

from numpy import linalg as LA

def adapt_SGD(f, f_gradient, X, Y, C, x0, epochs, L_0 = 0.0001, eps = 10, D_0 = 1, accel=False): #f  full
    time = timeit.default_timer()
    w_pred, L_pred = x0.copy(), L_0
    conv, t_conv = [], []
    w = w_pred.copy()
    conv.append(w)
    t_conv.append(timeit.default_timer()-time)
    j = 0
    for epoch in range(epochs):
        L_next = L_pred/4
        while True:
            L_next = L_next*2
            r = int(max(D_0//(L_next*eps),1))
            (batchX, batchY, j) = next_batch(X, Y, j, r)
            gradient = f_gradient(w_pred, batchX, batchY, C)
            w_next = w_pred - 1/(8*L_next)*gradient
            if f(w_next,batchX,batchY,C)<=f(w_pred,batchX,batchY,C)+gradient.dot(w_next-w_pred)+L_next*LA.norm(w_next-w_pred, 2)**2+eps/2:
                break
        w = w_next.copy()
        conv.append(w)
        t_conv.append(timeit.default_timer()-time)
        w_pred, L_pred = w_next, L_next
    return t_conv, conv[-1], conv

def adapt_SGDac(f, f_gradient, X, Y, C, x0, epochs, L_0 = 0.0001, eps = 10, D_0 = 10, accel=True, err = 0.1**8): #f  full
    time = timeit.default_timer()
    w_pred, L_pred = x0.copy(), L_0
    conv, fmin_pred, t_conv = [], float('inf'), []
    w = w_pred.copy()
    conv.append(w)
    t_conv.append(timeit.default_timer()-time)
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
            (batchX, batchY, j) = next_batch(X, Y, j, r)
            gradient = f_gradient(y, batchX, batchY, C)
            u_next = u_pred - 0.5*a*gradient
            w_next = (a*u_next+A_pred*w_pred)/A_next
            fmin_next = f(w_next,batchX,batchY,C)
            if fmin_next <= f(y,batchX,batchY,C)+gradient.dot(w_next-y)+L_next*LA.norm(w_next-y, 2)**2+a/(2*A_next)*eps:
                break 
        argmin = w_next if fmin_next < fmin_pred else w_pred
        w = w_next.copy()
        conv.append(w)
        t_conv.append(timeit.default_timer()-time)
        if abs(fmin_next - fmin_pred) < err: break
        w_pred, fmin_pred, L_pred, A_pred, u_pred = w_next, fmin_next, L_next, A_next, u_next,
    return t_conv, argmin, conv


def sol_quad(a,b,c):
    r = b**2 - 4*a*c
    if r > 0:
        return (-b + np.sqrt(r))/(2*a)
    elif r == 0:
        return (-b) / (2*a)
    else:
        print('non silution\n')
        raise ValueError

def V_task(F_grad, y, v_pred, beta, teta, C, Xb = None, yb = None, mu = None):
    if  not yb is None:
        mu = 1/X.shape[0] if mu==None else mu
        f1 = lambda x: f(x, Xb, yb)
        f1_grad = lambda x: f_gradient(x, Xb, yb)

        D_phi = lambda x,z: f1(x)  - f1(z) - f1_grad(z).dot(x-z) - mu/2*LA.norm(x-z,2)**2
        D_phi_grad = lambda x,z: f1_grad(x) - f1_grad(z) - mu*(x-z)
        V = lambda x: teta*(F_grad.dot(x)+g(x,C)) + (1-beta)*D_phi(x,v_pred) + beta*D_phi(x,y)
        V_grad = lambda x: teta*(F_grad+g_gradient(x,C)) + (1-beta)*D_phi_grad(x,v_pred) + beta*D_phi_grad(x,y)
    else:
        mu = 1/X.shape[0] if mu==None else mu
        f1 = lambda x, Xb, yb: f(x, Xb, yb)
        f1_grad = lambda x, Xb, yb: f_gradient(x, Xb, yb)    
        D_phi = lambda x,z, Xb, yb: f1(x, Xb, yb)  - f1(z, Xb, yb) - f1_grad(z, Xb, yb).dot(x-z) - mu/2*LA.norm(x-z,2)**2
        D_phi_grad = lambda x,z, Xb, yb: f1_grad(x, Xb, yb) - f1_grad(z, Xb, yb) - mu*(x-z)
        V = lambda x, Xb, yb, _: teta*(F_grad.dot(x)+g(x,C)) + (1-beta)*D_phi(x,v_pred, Xb, yb) + beta*D_phi(x,y, Xb, yb)
        V_grad = lambda x, Xb, yb, _: teta*(F_grad+g_gradient(x,C)) + (1-beta)*D_phi_grad(x,v_pred, Xb, yb) + beta*D_phi_grad(x,y, Xb, yb)
    return V, V_grad, D_phi
       
import liboptpy.base_optimizer as base
import numpy as np
import liboptpy.unconstr_solvers.fo as fo
import liboptpy.step_size as ss

def SPAG(f, f_gradient, g, g_gradient, X, Y, C, x0, epochs, L = 2, mu = 0.000001, solver = "Nesterov", err = 0.1**5):#f not full
    time = timeit.default_timer()
    sigm = 1/(1+mu/C[0])
    v_pred, w_pred, G_pred = x0, x0, 1
    conv, t_conv, fmin_pred = [], [], float('inf')
    w = w_pred.copy()
    conv.append(w)
    t_conv.append(timeit.default_timer()-time)
    A_pred, B_pred = 0, 1
    for epoch in range(epochs):
        G_next = max(1, G_pred/2)/2
        while True:
            if G_next > 10**10:
                print(f'G oversize on {len(conv)} iter')
                break
            G_next = 2*G_next
            a = sol_quad(L*G_next-sigm, -A_pred*sigm-B_pred, -A_pred*B_pred)
            A_next = A_pred + a
            B_next = B_pred + sigm*a
            alpha, beta, teta = a/A_next, a/B_next*sigm, a/B_next
            y = 1/(1-alpha*beta)*((1-alpha)*w_pred+alpha*(1-beta)*v_pred)
            
            mask = np.random.choice([True, False], size=X.shape[0], p=[0.0001, 0.9999])

            Xb, yb = X[mask], Y[mask]
            F_grad = f_gradient(y, Xb, yb)
            
            if solver == "Nesterov":
                V, V_grad, D_phi = V_task(F_grad, y, v_pred, beta, teta, C, Xb=Xb, yb=yb, mu = mu)
                adNest = fo.AcceleratedGD(V, V_grad, ss.Backtracking(rule_type="Lipschitz", rho=0.5, init_alpha=1))
                v_next = adNest.solve(x0=v_pred, max_iter=300, tol=0.1**4)
            elif solver == "my":
                V, V_grad, D_phi_ = V_task(F_grad, y, v_pred, beta, teta, C, mu = mu)
                D_phi = lambda tx,ty : D_phi_(tx, ty, Xb, yb)
                _, v_next, _ = adapt_SGDac(V, V_grad, Xb, yb, None, v_pred ,D_0 = 0.03*0.3, epochs = 120, err = 0.1**3)    
                
            w_next = (1-alpha)*w_pred+alpha*v_next
            w = w_next.copy()
            conv.append(w)
            t_conv.append(timeit.default_timer()-time)
            if D_phi(w_next, y) <= alpha**2 * G_next * ( (1-beta)*D_phi(v_next,v_pred) + beta*D_phi(v_next,y) ):
                break
        fmin_next = f(w_next, X, Y) + g(w_next, C)
        argmin = w_next if fmin_next < fmin_pred else w_pred
        if abs(fmin_next - fmin_pred) < err: break
        w_pred, fmin_pred, v_pred, G_pred, A_pred, B_pred = w_next, fmin_next, v_next, G_next, A_next, B_next
    return t_conv, argmin, conv








