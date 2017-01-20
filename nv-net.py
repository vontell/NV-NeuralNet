# A collection of useful methods which can be used to classify Nitrogen Vacancy
# Center spectrum scans
# Authors: Aaron Vontell and Elijah Stiles
# Date: January 17th, 2017

import scipy.io as sio
#import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution1D
from keras.optimizers import SGD
import argparse
import numpy as np
import os.path
import sys

EPOCHES = 50
NUM_XS = 2048
NUM_NEURONS = 2048
LEARNING_RATE = .001
MAX = 1500
MODEL_PATH = "./models/network.hdf5"


#def make_model():
#    
#    # Make the model here
#    model = Sequential()
#    model.add(Dense(output_dim=NUM_XS, input_dim=NUM_XS))
#    model.add(Activation("relu"))
#    model.add(Dense(output_dim=1))
#    model.add(Activation("sigmoid"))
#    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
#    return model

def make_model():
    
    # Make the model here
    model = Sequential()
    model.add(Dense(600, input_dim=100))
    model.add(Activation("relu"))
    model.add(Dropout(0.3))
    model.add(Dense(100))
    model.add(Activation("sigmoid"))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
    return model

def train_net(filename):
    print("Training neural net on " + filename + ".mat")
    data = load_mat_data(filename)

    # useful information for training
    good_vals = data["good"]
    bad_vals = data["bad"]
    
    data = []
    for val in good_vals:
        data.append((1, val[900:1000]))
    for val in bad_vals:
        data.append((0, val[900:1000]))
    np.random.shuffle(data)

    from keras.utils import np_utils
    X_train = np.array([np.array(i[1]) for i in data])
    Y_train = np.array([([0] if i[0] == 0 else [1]) for i in data])
    print X_train[0]
    print Y_train[0]
    
    model = make_model()
    model.fit(X_train, Y_train, nb_epoch=200, batch_size=5)
    
    for i in range(20):
        print(Y_train[i])
        print model.predict(np.reshape(X_train[i], (1, 100)))
    
#    X_train = []
#    Y_train = []
#    
#    for val in data:
#        X_train.append(np.reshape(val[1], (1, NUM_XS)))
#        classif = [[0, 1]] if val[0] == 1 else [[1, 0]]
#        Y_train.append(np.reshape(classif, (1,2)))
#    X_train = np.array(X_train)
#    Y_train = np.array(Y_train)
#    print X_train.shape
#    print Y_train.shape
#        
#    # Train the net here
#    model = make_model()
#    if os.path.isfile(MODEL_PATH):
#        pass#model.load_weights(MODEL_PATH)
#    
#    print(Y_train[0][0])
#    #plt.plot(range(2048), X_train[0][0])
#    #plt.show()
#    model.fit(X_train, Y_train, nb_epoch=EPOCHES, batch_size=NUM_XS)
#    
#    loss_and_metrics = model.evaluate(X_train[0], Y_train[0], batch_size=1)
#    print(loss_and_metrics)
#    
#    for val in data[0:10]:
#        x_val = np.reshape(val[1], (1, NUM_XS))
#        print(x_val)
#        classif = [[0, 1]] if val[0] == 1 else [[1, 0]]
#        prob = model.predict_proba(x_val, batch_size=NUM_XS)
#        print("Expected: " + str (classif) + "\tActual: " + str(prob))
#            
#    model.save_weights(MODEL_PATH)
#        
#    print("Finished training neural net, saved net to ", MODEL_PATH, ".hdf5")

def classify(filename, exclude):
    print("Classifying NV spectra from " + filename + ".mat")
    if exclude:
        print("Excluding this spectra from the training data")

    data = load_mat_data(filename)

    # useful information for training
    x_vals = data["x"]
    y_vals = np.transpose(data["y"])
    
    # Classify the data here
    
    return probs

def load_mat_data(filename):
    '''Loads data from a .mat file. Takes in the filename, without extension'''
    data = sio.loadmat(filename)
    return data


# Argument parser logic
parser = argparse.ArgumentParser()
parser.add_argument("--train", help="the filename (without .mat extension) to train on")
parser.add_argument("--classify", help="the filename (without .mat extension) to train on")
parser.add_argument('--omit', dest='omit', action='store_true',
                    help="include with --classify to omit this from the net training")
parser.set_defaults(omit=False)

args = parser.parse_args()
if args.train:
    train_net(args.train)
if args.classify:
    print(classify(args.classify, args.omit))