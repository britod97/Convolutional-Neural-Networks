# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np


def h(beta0, beta1, x):
    return(beta0 + beta1*x)
    
def calculateGradient(beta0, beta1, dataX, dataY):
    # Initialize the variables for the partial derivatives
    Dbeta0 = 0
    Dbeta1 = 0
    N = len(dataX)
    for k in range(N):
        # For every data entry we add (h(x) - y)/N
        Dbeta0 += (h(beta0, beta1, dataX[k]) - dataY[k])/N
        Dbeta1 += (h(beta0, beta1, dataX[k]) - dataY[k])*dataX[k]/N
    return(Dbeta0, Dbeta1)


def gradientDescent(beta0_0, beta1_0, dataX, dataY, alpha, eps, N_iter = 0):
    # If N_iter has a value of 0 we will simply iterate until convergence
    if N_iter == 0:
        distance = 1
        while distance > eps:
            Dbeta0, Dbeta1 = calculateGradient(beta0_0, beta1_0, dataX, dataY)
            
            beta0_1 = beta0_0 - alpha*Dbeta0
            beta1_1 = beta1_0 - alpha*Dbeta1
            
            distance = np.sqrt( (beta0_0 - beta0_1)**2 + (beta1_0 - beta1_1)**2  )
            beta0_0 = beta0_1
            beta1_0 = beta1_1
            
    # If N_iter has a value different from 0, we do the same as above but for a fixed number of iterations
    else:
        for iteration in range(N_iter):
            Dbeta0, Dbeta1 = calculateGradient(beta0_0, beta1_0, dataX, dataY)
            
            beta0_1 = beta0_0 - alpha*Dbeta0
            beta1_1 = beta1_0 - alpha*Dbeta1
            
            distance = np.sqrt( (beta0_0 - beta0_1)**2 + (beta1_0 - beta1_1)**2  )
            beta0_0 = beta0_1
            beta1_0 = beta1_1
    return(beta0_0, beta1_0)


if __name__ == '__main__':
    # read flash.dat to a list of lists
    datX = [i.strip().split() for i in open("data/ex2x.dat").readlines()]
    datY = [i.strip().split() for i in open("data/ex2y.dat").readlines()]

    data = [[float(L[0]) for L in datX], [float(L[0]) for L in datY]]
    
    beta0, beta1 = gradientDescent(0, 0, data[0], data[1], 0.07, 0.00001)
    
    X = np.linspace(2, 8, 100)
    Y = h(beta0, beta1, X)
    
    fig, ax = plt.subplots(1,1)
    ax.scatter(data[0], data[1])
    ax.plot(X, Y)
    ax.set_xlabel('Age in Years', fontsize=16)
    ax.set_ylabel('Height in meters', fontsize=16)
    plt.xticks(fontsize = 13)
    plt.yticks(fontsize=13)