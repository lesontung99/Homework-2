'''
    Template for polynomial regression
    AUTHOR Eric Eaton, Xiaoxiang Hu
'''

import numpy as np
from sklearn import preprocessing


#-----------------------------------------------------------------
#  Class PolynomialRegression
#-----------------------------------------------------------------

class PolynomialRegression:

    def __init__(self, degree, regLambda, init_theta):
        '''
        Constructor
        '''
        #self.omega = init_omega
        self.degree = degree
        self.regLambda = regLambda
        self.theta = init_theta
        self.mean = 0
        self.std = 0


        #TODO


    def polyfeatures(self, X, degree):
        '''
        Expands the given X into an n * d array of polynomial features of
            degree d.

        Returns:
            A n-by-d numpy array, with each row comprising of
            X, X * X, X ** 3, ... up to the dth power of X.
            Note that the returned matrix will not inlude the zero-th power.

        Arguments:
            X is an n-by-1 column numpy array
            degree is a positive integer
        '''
        #TODO
        n = len(X)
        Lala = np.transpose([X])
        #print(Lala)
        orgx = Lala
       
        for count in range (degree):
            pluspoly = np.ones((n,1))
            for mul in range (count+2):
                pluspoly = pluspoly*orgx
            Lala = np.concatenate((Lala,pluspoly), axis=1)
       
        #print(Lala)
        return Lala



        

    def fit(self, X, y,):
        '''
            Trains the model
            Arguments:
                X is a n-by-1 array
                y is an n-by-1 array
            Returns:
                No return value
            Note:
                You need to apply polynomial expansion and scaling
                at first
        '''
        
        Xex = PolynomialRegression.polyfeatures(self, X, degree = self.degree)
        n,d = Xex.shape
        Xex = np.concatenate((np.ones((n,1)),Xex), axis = 1)
        
        mean = Xex.mean(axis=0)
        std = Xex.std(axis=0)
        X = (Xex - mean) / std
        X = np.nan_to_num(X,True,0,9999,-9999)
        self.mean = mean
        self.std = std
        
        #I believe something is wrong with this. Are you sure it need to be standard scaler?
        
        # construct reg matrix
        regMatrix = self.regLambda * np.eye(d +1 )
        regMatrix[0,0] = 0

        # analytical solution (X'X + regMatrix)^-1 X' y
        self.theta = np.linalg.pinv(X.T.dot(X) + regMatrix).dot(X.T).dot(y)
        #I'd rather use linear regression.
        '''
        for i in range (1000):
            error = Xex@self.theta - y
            plus = Xex.T.dot(error)
            self.theta = self.theta - plus*(self.regLambda/1000)

        '''
        #print("Theta:")
        #print(self.theta)
        #TODO
        
        
    def predict(self, X):
        '''
        Use the trained model to predict values for each instance in X
        Arguments:
            X is a n-by-1 numpy array
        Returns:
            an n-by-1 numpy array of the predictions
        '''
        # TODO
        
        X = PolynomialRegression.polyfeatures(self, X, degree = self.degree)
        n,d = X.shape
        Xex = np.concatenate((np.ones((n,1)),X), axis = 1)
        mean = self.mean
        std = self.std
        X = (Xex - mean) / std
        X = np.nan_to_num(X,True,0,9999,-9999)
        newy = self.theta.dot(X.T)
        newy = newy*std + mean
        #print(self.theta)
        
        # predict
        return newy



#-----------------------------------------------------------------
#  End of Class PolynomialRegression
#-----------------------------------------------------------------


def learningCurve(Xtrain, Ytrain, Xtest, Ytest, regLambda, degree):
    '''
    Compute learning curve
        
    Arguments:
        Xtrain -- Training X, n-by-1 matrix
        Ytrain -- Training y, n-by-1 matrix
        Xtest -- Testing X, m-by-1 matrix
        Ytest -- Testing Y, m-by-1 matrix
        regLambda -- regularization factor
        degree -- polynomial degree
        
    Returns:
        errorTrains -- errorTrains[i] is the training accuracy using
        model trained by Xtrain[0:(i+1)]
        errorTests -- errorTrains[i] is the testing accuracy using
        model trained by Xtrain[0:(i+1)]
        
    Note:
        errorTrains[0:1] and errorTests[0:1] won't actually matter, since we start displaying the learning curve at n = 2 (or higher)
    '''
    
    n = len(Xtrain)
    
    errorTrain = np.zeros((n))
    errorTest = np.zeros((n))
    for i in range(2, n):
        Xtrain_subset = Xtrain[:(i+1)]
        Ytrain_subset = Ytrain[:(i+1)]
        init_theta = np.ones((n,1))
        model = PolynomialRegression(degree, regLambda, init_theta=init_theta)
        model.fit(Xtrain_subset,Ytrain_subset)
        
        predictTrain = model.predict(Xtrain_subset)
        err = predictTrain - Ytrain_subset
        errorTrain[i] = np.multiply(err, err).mean()
        
        predictTest = model.predict(Xtest)
        err = predictTest - Ytest
        errorTest[i] = np.multiply(err, err).mean()
    
    return (errorTrain, errorTest)