# NV-NeuralNet

These Python scripts can be used to load .MAT files with NV spectrum data from Nitrogen-Vacancy scans, and classify whether or not it is actually a NV-Center (with a certain probability). This is acheived using a multi-layer perceptron and TensorFlow.

### Installation Steps
To setup this software, simply run the following command:

```
sudo pip install -r requirements.txt
```

This will install `scipy` for MATLAB interaction, and Tensor Flow for the actual Neural Net.

### Usage Instructions
Usage of this software can also be seen with `python nv-net.py -h`, but here are some more detailed instructions:

#### Training the Neural Net
To train the neural net on an existing .MAT file, use the following command, where `filename` is the name of the .MAT file you wish to read from (without the file extension .MAT):

```
python nv-net.py --train filename
```

Note that this file must have the following data structure:

* `x` - 1xL vector which applies to all y values
* `good` - NxL y vectors of good spectra which come from an NV-center
* `bad` - MxL y vectors of bad spectra which come from something other than an NV

This essentially provides us with negative and positive samples which we can use to train our neural net. Note that you can provide a different number of good and bad samples, but the number of data points must be the same in each (for example, in `training_data.mat`, L = 2048)

#### Classifying a Spectra Sample
After training the neural net, you can provide a .MAT file with the spectra data from an NV scan, which will return a single value [0.0, 1.0] which will be the probability that the given sample is characteristic of an NV center. Use this command in Python:

```
python nv-net.py --classify filename -omit
```

Note that `filename` is the name of the file with the spectra data without the .MAT extension. Note that the .MAT file must have the following structure. The `-omit` flag should be used if you wish to omit this example from the trained neural net (i.e. if no omit flag is given, the neural net will train itself with this new sample).

* `x` - 1xL vector which applies to the y values
* `y` - 1xL vector which are the y values of the spectra sample

Note that L must be the same as L in the training data.

#### Using in MATLAB

Coming soon...
