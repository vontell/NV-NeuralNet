# A collection of useful methods which can be used to classify Nitrogen Vacancy
# Center spectrum scans
# Authors: Aaron Vontell and Elijah Stiles
# Date: January 17th, 2017

import tensorflow as tf
import scipy.io as sio
#import matplotlib.pyplot as plt
import argparse
import numpy as np

EPOCHES = 100
NUM_XS = 2048
NUM_NEURONS = 512
LEARNING_RATE = .001
MODEL_PATH = "./models/network.txt"


def make_model():
    weights = {
        'wf1': tf.Variable(tf.random_normal([NUM_XS, NUM_NEURONS], stddev=np.sqrt(2. / NUM_XS))),
        'wf2': tf.Variable(tf.random_normal([NUM_NEURONS, NUM_NEURONS], stddev=np.sqrt(2. / NUM_NEURONS))),
        'wf3': tf.Variable(tf.random_normal([NUM_NEURONS, NUM_NEURONS], stddev=np.sqrt(2. / NUM_NEURONS))),
        'wo': tf.Variable(tf.random_normal([NUM_NEURONS, 2], stddev=np.sqrt(2. / NUM_NEURONS)))
    }
    biases = {
        'bf1': tf.Variable(tf.zeros(NUM_NEURONS)),
        'bf2': tf.Variable(tf.zeros(NUM_NEURONS)),
        'bf3': tf.Variable(tf.zeros(NUM_NEURONS)),
        'bo': tf.Variable(tf.zeros(2))
    }

    inputs = tf.placeholder(tf.float32, [None, NUM_XS])
    actual_output = tf.placeholder(tf.float32, [None, 2])

    # Layer 1
    fc1 = tf.add(tf.matmul(inputs, weights['wf1']), biases['bf1'])
    fc1 = tf.nn.relu(fc1)

    # Layer 2
    fc2 = tf.add(tf.matmul(fc1, weights['wf2']), biases['bf2'])
    fc2 = tf.nn.relu(fc2)

    # Layer 3
    fc3 = tf.add(tf.matmul(fc2, weights['wf3']), biases['bf3'])
    fc3 = tf.nn.relu(fc3)

    # Output Layer
    out = tf.add(tf.matmul(fc3, weights['wo']), biases['bo'])
    out = tf.nn.softmax(out)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(actual_output * tf.log(out), reduction_indices=[1]))
    train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)
    return out, train_op, inputs, actual_output


def train_net(filename):
    print("Training neural net on " + filename + ".mat")
    data = load_mat_data(filename)

    # useful information for training
    x_vals = data["x"][0]
    good_vals = data["good"][0:10]
    bad_vals = data["bad"][0:10]

    out, train_op, inputs, actual_output = make_model()
    saver = tf.train.Saver()
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(EPOCHES):
            for inp in good_vals:
                inp = np.reshape(inp, (1, NUM_XS))
                sess.run(train_op, feed_dict={inputs: inp, actual_output: [[0, 1]]})
            for inp in bad_vals:
                inp = np.reshape(inp, (1, NUM_XS))
                sess.run(train_op, feed_dict={inputs: inp, actual_output: [[1, 0]]})
        
        saver.save(sess, MODEL_PATH)
		
	print("Finished training neural net")


def classify(filename, exclude):
    print("Classifying NV spectra from " + filename + ".mat")
    if exclude:
        print("Excluding this spectra from the training data")

    data = load_mat_data(filename)

    # useful information for training
    x_vals = data["x"]
    y_vals = np.transpose(data["y"])

    probs = []

    out, train_op, inputs, actual_output = make_model()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, MODEL_PATH)
        sample_prob = sess.run(out, feed_dict={inputs: y_vals})
        print(sample_prob)
        probs.append(sample_prob[1])

    #plt.plot(x_vals, y_vals)
    #plt.show()
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
    classify(args.classify, args.omit)
