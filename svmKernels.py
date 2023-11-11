"""
Custom SVM Kernels

Author: Eric Eaton, 2014

"""

import numpy as np


_polyDegree = 2
_gaussSigma = 25


def myPolynomialKernel(X1, X2):
    '''
        Arguments:  
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    '''
    kernel = ((X1.dot(X2.T))+1)
    kernel = np.power(kernel, _polyDegree)
    return kernel



def myGaussianKernel(X1, X2):
    '''
        Arguments:
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    '''
    # Since I don't knows how to combine this all into matrix form
    n1,d1 = X1.shape
    n2,d2 = X2.shape
    print (_gaussSigma)
    kernel = np.zeros((n1,n2))
    for i in range (n1):
        '''
        for j in range (n2):
            v1 = X1[i,:]
            v2 = X2[j,:]
            kernel[i,j] = np.exp(np.linalg.norm(v1 - v2)/(2*_gaussSigma))
            # Too slow.
        '''
        v1 = X1[i,:]
        v2 = X2 - v1
        v2 = v2*v2
        rowi = v2.sum(axis = 1)
        kernel[i,:] = np.exp(-rowi/(2*_gaussSigma))
    return kernel



def myCosineSimilarityKernel(X1,X2):
    '''
        Arguments:
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    '''
    return #TODO (CIS 519 ONLY)

