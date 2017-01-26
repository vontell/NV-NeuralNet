# NV-NeuralNet

These Python scripts can be used to load .MAT files with NV spectrum data from Nitrogen-Vacancy scans, and classify whether or not it is actually a NV-Center (with a certain probability). This is acheived using a multi-layer perceptron and TensorFlow.

**Most recent results obtain an accuracy of 92.11%**

### Installation Steps
To setup this software, simply run the following command:

```
sudo pip install -r requirements.txt
```

This will install `numpy`, `scipy` for MATLAB interaction, and `tensorflow` for the actual Neural Network. *Note that some operating systems may require [special installation instructions for TensorFlow](https://www.tensorflow.org/get_started/os_setup).*

### Usage Instructions
Usage of this software can also be seen with `python nv-net.py -h`, but here are some more detailed instructions:

#### Training the Neural Net
To train the neural net on an existing .MAT file, use the following command, where `filename` is the name of the .MAT file you wish to read from (**without the file extension .MAT**):

```
python nv-net.py --train filename
```

Note that this file must have the following data structure:

* `x` - 1xL vector which applies to all y values
* `good` - NxL y vectors of good spectra which come from an NV-center
* `bad` - MxL y vectors of bad spectra which come from something other than an NV

This essentially provides us with negative and positive samples which we can use to train our neural net. Note that you can provide a different number of good and bad samples, but the number of data points must be the same in each (for example, in `training_data.mat`, L = 2048)

Two additional flags exit for the training operation, `--overwrite` and `--graph`. If you would like to overwrite any existing saved model which was previously trained, you should include `--overwrite`; otherwise no new training will occur. If you would like to display relevant neural network and training information after training, include the `--graph` flag. Both of these flags default to false.

For example, to train a new network from `filename`, display training graphs at the end, and overwrite an existing network, use the following command:

```
python nv-net.py --train filename --overwrite --graph
```

#### Classifying a Spectra Sample
After training the neural net, you can provide a .MAT file with the spectra data from an NV scan, which will return a single value (0 or 1) which will be indicate if the given scan is characteristic of an NV (0 if it is believed to not have come from an NV, and 1 if it is believed that the scan shows an NV). You can run this Python command with:

```
python nv-net.py --classify filename
```

By default, the sample being classified is added to the neural net training data. The `--omit` flag should be used if you wish to exclude this example from the trained neural net (i.e. if no omit flag is given, the neural net will train itself with this new sample):

```
python nv-net.py --classify filename --omit
```

Note that `filename` is the name of the file with the spectra data without the .MAT extension. Also note that the .MAT file must have the following structure:

* `x` - 1xL vector which applies to the y values
* `y` - 1xL vector which are the y values of the spectra sample

Note that L must be the same as L in the training data.

#### Using in MATLAB

Coming soon...
