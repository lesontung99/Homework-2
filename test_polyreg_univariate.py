'''
    TEST SCRIPT FOR POLYNOMIAL REGRESSION 1
    AUTHOR Eric Eaton, Xiaoxiang Hu
'''

import numpy as np
import matplotlib.pyplot as plt
from polyreg import PolynomialRegression

if __name__ == "__main__":
    '''
        Main function to test polynomial regression
    '''

    # load the data
    filePath = "data\polydata.dat"
    file = open(filePath,'r')
    allData = np.loadtxt(file, delimiter=',')

    X = allData[:, 0]
    y = allData[:, 1]
    print("x = ")
    print(X)
    print("y = ")
    print(y)
    # regression with degree = d
    n = len(X)
    init_theta = np.ones((n,1))
    d = 8
    regLambda = 0
    model = PolynomialRegression(degree = d, regLambda = regLambda, init_theta = init_theta)
    model.fit(X, y)
    
    # output predictions
    xpoints = np.linspace(np.min(X), np.max(X), 100).T
    ypoints = model.predict(xpoints)
    print(xpoints)
    print(ypoints)
    # plot curve
    plt.figure()
    plt.plot(X, y, 'rx')
    plt.title('PolyRegression with d = '+str(d)+' and Lambda = '+str(regLambda) )
    #plt.hold(True)
    plt.plot(xpoints, ypoints, 'b-')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

