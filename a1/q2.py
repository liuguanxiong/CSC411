# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 20:39:09 2017

"""
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
import scipy.misc as sm
import math
np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506,1)),x),axis=1) #add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))

#helper function
def l2(A,B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (B**2).sum(axis=1).reshape(1,B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist

#helper function
def run_on_fold(x_test, y_test, x_train, y_train, taus):
    '''
    Input: x_test is the N_test x d design matrix
           y_test is the N_test x 1 targets vector        
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           taus is a vector of tau values to evaluate
    output: losses a vector of average losses one for each tau value
    '''
    N_test = x_test.shape[0]
    losses = np.zeros(taus.shape)
    for j,tau in enumerate(taus):
        predictions =  np.array([LRLS(x_test[i,:].reshape(d,1),x_train,y_train, tau) \
                        for i in range(N_test)])
        losses[j] = ((predictions.flatten()-y_test.flatten())**2).mean()
    return losses
 
 
#to implement
def LRLS(test_datum,x_train,y_train, tau,lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''

    numerator = np.exp(-(l2(test_datum.transpose(), x_train))/(2*tau**2))
    denominator = np.exp(sm.logsumexp(-(l2(test_datum.transpose(), x_train))/(2*tau**2)))
    a_i = numerator/denominator
    a_i = a_i.flatten()
    A = np.diag(a_i)

    XtAX = np.dot(np.dot(np.transpose(x_train), A),x_train)
    XtAy = np.dot(np.dot(np.transpose(x_train), A),y_train)
    W = np.linalg.solve(XtAX + (lam*np.identity(XtAX.shape[0])), XtAy)

    return np.dot(W,test_datum)[0]

def run_k_fold(x,y,taus,k):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector
           taus is a vector of tau values to evaluate
           K in the number of folds
    output is losses a vector of k-fold cross validation losses one for each tau value
    '''

    N = x.shape[0]
    foldsize = int(np.round(N/k))
    losses = np.zeros(len(taus))
    indices = np.array_split(idx,k)
    for i in range(k):
        print("fold: ",i+1)
        train_index = [j for j in idx if j not in indices[i]]
        x_test = np.take(x, indices[i], 0)
        x_train = np.take(x, train_index, 0)
        y_test = np.take(y, indices[i], 0)
        y_train = np.take(y, train_index, 0)
        losses = losses + run_on_fold(x_test, y_test, x_train, y_train, taus)
    return losses/k


if __name__ == "__main__":
    # In this exercise we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0,3,200)
    losses = run_k_fold(x,y,taus,k=5)
    plt.plot(taus,losses)
    ax = plt.gca()
    ax.grid()
    plt.xlabel("Taus")
    plt.ylabel("Average Loss")
    plt.savefig("Loss.png")
    print("min loss = {}".format(losses.min()))
