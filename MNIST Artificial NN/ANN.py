# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def logistic(Z):
    Z = np.array(Z)
    return(1/(1+np.exp(-Z)))

def logisticDeriv(Z):
    return(logistic(Z)*(1-logistic(Z)))
    
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return(one_hot_Y)

def forwardprop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = logistic(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = logistic(Z2)
    return Z1, A1, Z2, A2

def backprop(A1, Z1, A2, Z2, W1, W2, X, Y, m):
    dZ2 = A2 - one_hot(Y)
    dW2 = (1/m)*(dZ2.dot(A1.T))
    db2 = (1/m)*np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2)*logisticDeriv(Z1)
    dW1 = (1/m)*dZ1.dot(X.T)
    db1 = (1/m)*np.sum(dZ1)
    return dW2, db2, dW1, db1

def nextIter(W1, b1, W2, b2, db1, dW1, db2, dW2, alpha):
    W1 = W1 - alpha*dW1
    b1 = b1 - alpha*db1
    W2 = W2 - alpha*dW2
    b2 = b2 - alpha*db2
    return W1, b1, W2, b2

def makePredictions(W1, b1, W2, b2, X):
    _, _, _, predictions = forwardprop(W1, b1, W2, b2, X)
    predictions = predictions.T
    return(predictions)

def calculateError(predictions, Y):
    m = predictions.shape[0]
    Y = one_hot(Y).T
    error = (-1/m)*np.sum(Y*(np.log(predictions)) + (1-Y)*np.log(1-predictions))
    return(error)

def calculateAccuracy(predictions, Y):
    m = predictions.shape[0]
    L = [Y[i] == np.argmax(predictions[i]) for i in range(len(predictions))]
    return(np.sum(L)/m)
    

def gradDescent(W1, W2, b1, b2, X, Y, alpha, N_iter):
    m, n = X.shape
    error_per_epoch = []
    acc_per_epoch = []
    for epoch in range(N_iter):
        predictions = makePredictions(W1, b1, W2, b2, X_test)
        E = calculateError(predictions, Y_test)
        A = calculateAccuracy(predictions, Y_test)
        error_per_epoch.append(E)
        acc_per_epoch.append(A)
        
        Z1, A1, Z2, A2 = forwardprop(W1, b1, W2, b2, X)
        dW2, db2, dW1, db1 = backprop(A1, Z1, A2, Z2, W1, W2, X, Y, m)
        W1, b1, W2, b2 = nextIter(W1, b1, W2, b2, db1, dW1, db2, dW2, alpha)
        
    return(W1, b1, W2, b2, error_per_epoch, acc_per_epoch)


if __name__ == '__main__':
    data = pd.read_csv('mnist.txt', sep=' ', header=None)   # Read text file
    del data[0]                                             # Remove first column
    
    data.columns = list(data.columns)[:-1] + ['digitclass'] # Change name of last column to 'digitclass'
    data = data[:1000]                                      # Take only the first 1000 entries
    
    X = np.array(data.drop('digitclass', axis=1))/255
    X[0][315] = 0
    Y = np.array(data['digitclass'])
    X_train, X_test = X[:900].T, X[900:].T
    Y_train, Y_test = Y[:900], Y[900:]
    
    A = []
    J = []
    iterations = 2000
    maxi = 20
    alpha = 0.1
    
    for i in range(maxi):
        W1 = np.random.rand(20, 784)-0.5
        W2 = np.random.rand(10, 20)-0.5
        b1 = np.random.rand(20,1)-0.5
        b2 = np.random.rand(10,1)-0.5
        
        W1, b1, W2, b2, error, acc = gradDescent(W1, W2, b1, b2, X_train, Y_train, alpha, iterations)
        
        print("Run number:",i, ". Iterations:",iterations, ". Alpha:",alpha)
        print("Final accuracy:", acc[-1], "Final cost:", error[-1])
        print("---------------------")
        
        A.append(acc[-1])
        J.append(error[-1])
        
    print("______________________________")
    print("Average accuracy obtained after", maxi, "runs, with", iterations, "iterations:", np.average(A))
    print("Average cost obtained after", maxi, "runs, with", iterations, "iterations:", np.average(J))