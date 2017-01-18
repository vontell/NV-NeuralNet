# A collection of useful methods which can be used to classify Nitrogen Vacancy
# Center spectrum scans
# Authors: Aaron Vontell and Elijah Stiles
# Date: January 17th, 2017

import scipy.io as sio
import matplotlib.pyplot as plt
import argparse

def train_net(filename):
	print("Training neural net on " + filename + ".mat")
	data = load_mat_data(filename)
	
	# useful information for training
	x_vals = data["x"][0]
	good_vals = data["good"]
	bad_vals = data["bad"]
	
	plt.plot(x_vals, good_vals[60])
	plt.show()
	
def classify(filename, exclude):
	print("Classifying NV spectra from " + filename + ".mat")
	if exclude:
		print("Excluding this spectra from the training data")
		
	data = load_mat_data(filename)
	
	# useful information for training
	x_vals = data["x"]
	y_vals = data["y"]
	
	plt.plot(x_vals, y_vals)
	plt.show()
	
def load_mat_data(filename):
	'''Loads data from a .mat file. Takes in the filename, without extension'''
	data = sio.loadmat(filename)
	return data

# Argument parser logic
parser = argparse.ArgumentParser()
parser.add_argument("--train", help="the filename (without .mat extension) to train on")
parser.add_argument("--classify", help="the filename (without .mat extension) to train on")
parser.add_argument('--omit', dest='omit', action='store_true', help="include with --classify to omit this from the net training")
parser.set_defaults(omit=False)

args = parser.parse_args()
if args.train:
    train_net(args.train)
if args.classify:
	classify(args.classify, args.omit)