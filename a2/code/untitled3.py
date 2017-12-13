#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 01:01:59 2017

@author: Guanxiong
"""

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array
    '''
    rval = np.zeros((digits.shape[0],10))
    d = digits.shape[1]
    for k in range(0,10):

        SigmaK_det = np.linalg.det(covariances[k])
        Sigma_inv = np.linalg.inv(covariances[k])
        constant = (-d/2)*np.log(2*np.pi) + (-1/2)*np.log(SigmaK_det)

        for idx in range(0,digits.shape[0]):

            # deal with matrix multiplication
            minus = np.subtract(digits[idx],means[k]).reshape(1, d)
            minusT = minus.transpose().reshape(d, 1)
            val = np.matmul(np.matmul(minus, Sigma_inv), minusT) * (-1/2)
            rval[idx, k] = val + constant

    return rval


def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    # log p(y|x, eta) = log p(x|y, eta) + log p(y) - log p(x)
    generative = generative_likelihood(digits, means, covariances)
    log_px = np.log(np.divide(np.sum(np.exp(generative),axis=1), 10))
    return (generative + np.log(0.1)) - log_px.reshape(-1,1)