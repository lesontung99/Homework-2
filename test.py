"""
=====================================
Test SVM using GridSearch
=====================================

Author: Eric Eaton, 2014

Adapted from scikit_learn documentation.

"""
print(__doc__)
from numpy import loadtxt
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
#from svmKernels import myGaussianKernel
#from svmKernels import _gaussSigma
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import pairwise_distances
from sklearn.pipeline import Pipeline
#from sklearn.metrics.pairwise import chi2_kernel
from sklearn.metrics import accuracy_score



def myGaussianKernel(X1, X2,gamma):
        '''
            Arguments:
                X1 - an n1-by-d numpy array of instances
                X2 - an n2-by-d numpy array of instances
            Returns:
                An n1-by-n2 numpy array representing the Kernel (Gram) matrix
        '''
        # Since I don't knows how to combine this all into matrix form
        '''
        n1,d1 = X1.shape
        n2,d2 = X2.shape
        print(_gaussSigma)
        kernel = np.zeros((n1,n2))
        for i in range (n1):
            
            for j in range (n2):
                v1 = X1[i,:]
                v2 = X2[j,:]
                kernel[i,j] = np.exp(np.linalg.norm(v1 - v2)/(2*_gaussSigma))
                # Too slow.
            
            v1 = X1[i,:]
            v2 = X2 - v1
            v2 = v2*v2
            rowi = v2.sum(axis = 1)
            kernel[i,:] = np.exp(-rowi/(2*_gaussSigma))
            
        '''
        # Cusmtom Kernel too slow. SOrry.
        kernel = pairwise_distances(X1,X2)
        kernel = np.exp(-(kernel*kernel)/(2*gamma))
        return kernel
    # import some data to play with

class CustomKernel(BaseEstimator,TransformerMixin):
    def __init__(self, gamma=1.0):
        super(CustomKernel,self).__init__()
        self.gamma = gamma

    def transform(self, X):
        return myGaussianKernel(X, self.X_train_,gamma=self.gamma)

    def fit(self, X, y=None, **fit_params):
        self.X_train_ = X
        return self

if __name__ == "__main__":
    filename = 'Questions\CIS419-master\Assignment2\hw2_skeleton\data\svmTuningData.dat'
    data = loadtxt(filename, delimiter=',')
    X = data[:, 0:2]
    Y = data[:, 2]

    print ("Training the SVMs...")

    #C = 1.0  # value of C for the SVMs
    #_gaussSigma = 0.5
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.66, shuffle=True)

    initList = np.array([1,3,6,8])
    testList = np.array([0.5])
    gammaTestList = testList
    for i in range(-5,5):
        addList = initList*np.power(10.0,i)
        testList = np.concatenate((testList,addList))
        if i <= 3:
            gammaTestList = np.concatenate((gammaTestList,addList))
    #gamma = 0.5
    pipe = Pipeline(
        [
            ('custom', CustomKernel()),
            ('svm', svm.SVC())
        ]
    )
    parameter = dict(
         [
              ('custom__gamma', gammaTestList),
              ('svm__kernel',['precomputed']),
              ('svm__C', testList)
         ]
    )
    # create an instance of SVM with the custom kernel and train it
    #myModel = svm.SVC(C = C, kernel=build_Gaussian(gamma))

    clf = GridSearchCV(pipe,param_grid=parameter,n_jobs=-1,error_score='raise',verbose=1)
    clf.fit(X_train,Y_train)

    '''
    # create an instance of SVM with build in RBF kernel and train it
    equivalentGamma = 1.0 / (2 * _gaussSigma ** 2)
    model = svm.SVC(C = C, kernel='rbf', gamma=equivalentGamma)
    model.fit(X, Y)
    '''
    
    print ("")
    print ("Testing the SVMs...")

    h = .02  # step size in the mesh

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Y_pred = clf.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    print ("Accuracy = ", accuracy)
    # get predictions for both my model and true model
    myPredictions = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    myPredictions = myPredictions.reshape(xx.shape)

    '''
    predictions = model.predict(np.c_[xx.ravel(), yy.ravel()])
    predictions = predictions.reshape(xx.shape)
    '''

    # plot my results
    #plt.subplot(1, 2, 1)
    plt.pcolormesh(xx, yy, myPredictions, cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], c=Y,edgecolors='k', cmap=plt.cm.Paired) # Plot the training points
    plt.title("SVM with Best param: (sigma = "+str(clf.best_params_) +")")
    plt.axis('tight')

    # plot built-in results
    '''
    plt.subplot(1, 2, 2)
    plt.pcolormesh(xx, yy, predictions, cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k',cmap=plt.cm.Paired) # Plot the training points
    plt.title('SVM with Equivalent Scikit_learn RBF Kernel for Comparison')
    plt.axis('tight')
    '''
    plt.show()
