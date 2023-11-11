'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton
'''

import numpy as np
from scipy.special import expit

class LogisticRegression:

    def __init__(self, alpha = 0.01, regLambda=0.01, epsilon=0.0001, maxNumIters = 10000):
        '''
        Constructor
        '''
        self.alpha = alpha
        self.regLambda = regLambda
        self.epsilon = epsilon
        self.maxNumIter = maxNumIters
        self.currentIter = 0
        self.theta = None

    

    def computeCost(self, theta, X, y, regLambda):
        '''
        Computes the objective function
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            regLambda is the scalar regularization constant
        Returns:
            a scalar value of the cost  ** make certain you're not returning a 1 x 1 matrix! **
        '''
        offset = np.sqrt(theta.T.dot(theta))*regLambda/2
        mainSum = 0
        n,d = X.shape
        epsilon = 1E-6
        x = np.concatenate((np.ones((n,1)),X), axis=1)
        for k in range (n):
            xij = x[k,:]
            xi = xij[:,np.newaxis]
            dim1 = xi.shape
          
            ult = LogisticRegression.sigmoid(self,xi)
            
            yi = y[k]
            dim = ult
            dim = -yi*np.log(dim+ epsilon) - (1 - yi)*np.log(1 - dim + epsilon)
            #print(dim)
            
            mainSum += dim
        totalSum = offset+mainSum
        return totalSum

    
    
    def computeGradient(self, theta, X, y, regLambda):
        '''
        Computes the gradient of the objective function
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            regLambda is the scalar regularization constant
        Returns:
            the gradient, an d-dimensional vector
        '''
        #offset = regLambda
        #mainSum = 0
        n,d = X.shape
        Grad = np.zeros((d,1))
        ur = LogisticRegression.sigmoid(self,X.T)
        #x = np.concatenate((np.ones(n,1),X), axis=1)
        for k in range (d):
            xij = X[:,k]
            xi = xij[:, np.newaxis]
            if k == 0:
                Grad[k] = (ur - y.T).dot(xi)
            else:
                Grad[k] = (ur - y.T).dot(xi)+regLambda*theta[k]
            
        return Grad
    


    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
        '''
        n,d = X.shape
        x = np.concatenate((np.ones((n,1)),X), axis = 1)
        d = d + 1
        thetaplus = LogisticRegression.init_theta(d)
        theta = thetaplus[:,np.newaxis]
        self.theta = theta
        cost = LogisticRegression.computeCost(self, theta,X,y,self.regLambda)
        print("theta:\n", self.theta)
        print("cost:", cost)
        print("AFter regression:")
        while LogisticRegression.converged(self, x, y, theta):
            lastTheta = theta
            error = LogisticRegression.computeGradient(self, theta, x,y, self.regLambda)*self.alpha
            theta = theta - error
            self.theta = lastTheta
            self.currentIter += 1
            
        cost = LogisticRegression.computeCost(self, theta,X,y,self.regLambda)
        print("cost:", cost)
        print("theta:\n", self.theta)
        print(self.currentIter)







    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy matrix
        Returns:
            an n-dimensional numpy vector of the predictions
        '''
        n,d = X.shape
        print (n,d)
        x = np.concatenate((np.ones((n,1)),X), axis = 1)
        apha = LogisticRegression.sigmoid(self,x.T)
        return apha
        



    def sigmoid(self, Z):
        apha = self.theta.T.dot(Z)
        #beta = apha[0]
        g = expit(apha)
       
        return g
    	
    def converged(self, X, y, theta):
        if self.currentIter<=10:
            return True
        old = X.dot(self.theta) - y
        new = X.dot(theta) - y
        offset = new - old
        #if (self.currentIter%1000)==0:
            #print(offset)
        # Check for convergence
        if self.currentIter >= self.maxNumIter:
            return False
        elif np.sqrt(offset.T.dot(offset)) <= self.epsilon:
            return False
        else:
            return True
    
    def init_theta(n):
        x = np.random.normal(size=n)
        x -= x.mean()
        return x / np.linalg.norm(x)
