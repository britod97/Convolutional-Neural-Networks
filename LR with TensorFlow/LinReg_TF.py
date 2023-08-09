# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping


def readData(path):
    return(np.array([float(i.strip().split()[0]) for i in open(path).readlines()]))

def getWeights():
    theta1 = model.get_weights()[0][0][0]
    theta0 = model.get_weights()[1][0]
    return(theta1, theta0)

if __name__ == '__main__':
    
    # First we read the data:
    x_data = readData("data/ex2x.dat")
    y_data = readData("data/ex2y.dat")
    length_x = len(x_data)
    
    # Set seeds for the random number generators
    tf.keras.utils.set_random_seed(100)
    np.random.seed(100)
    
    # Create the model:
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1, input_shape=[1]))
    model.compile(loss ='mean_squared_error', optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=0.01))
    
    # Get initial weights of the parameters
    theta1, theta0 = getWeights()
    print("Initial values:", "theta1 = ", theta1, ", theta0 = ", theta0)
     
    # Fit the model and get the parameter weights again
    history = model.fit(x_data, y_data, epochs=5000, verbose=0)
    theta1 = model.get_weights()[0][0][0]
    theta0 = model.get_weights()[1][0]
    print("Final values:", "theta1 = ", theta1, ", theta0 = ", theta0)
    
    # To plot the line, use model.predict over the x values
    predictions = model.predict(x_data)
    
    # Make the plot
    fig, ax = plt.subplots(1,1)
    ax.scatter(x_data, y_data)
    ax.plot(x_data, predictions)
    ax.set_xlabel('Age in Years', fontsize=16)
    ax.set_ylabel('Height in meters', fontsize=16)
    plt.xticks(fontsize = 13)
    plt.yticks(fontsize=13)