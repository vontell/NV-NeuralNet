
import tensorflow as tf
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import math
import time
from display_utils import DynamicConsoleTable
from sound_utils import Sounds

###############################################################################

num_kernels = 1

def model(learning_rate=0.01, dropout=1.0):
    # Parameters
    weights = {
        'conv1': tf.Variable(tf.random_normal([3, 1, 1, num_kernels], stddev=np.sqrt(2./(3*1*1)))),
        'out': tf.Variable(tf.random_normal([100*num_kernels, 2], stddev=np.sqrt(2./(100*num_kernels)))),
    }
    biases = {
        'conv1': tf.Variable(tf.zeros(num_kernels)),
        'out': tf.Variable(tf.zeros(2)),
    }

    # Placeholders for training data
    x = tf.placeholder(tf.float32, [None, 100, 1, 1])
    y = tf.placeholder(tf.int64, [None])

    # Input -> Conv + ReLU
    conv1 = tf.nn.conv2d(x, weights['conv1'], strides=[1, 1, 1, 1], padding='SAME')
    conv1 = tf.nn.relu(tf.nn.bias_add(conv1, biases['conv1']))
    lrn1 = tf.nn.local_response_normalization(conv1, depth_radius=5, bias=1.0, alpha=1e-4, beta=0.75)
    lrn1 = tf.nn.dropout(lrn1, dropout)
    
    # FC -> Output FC
    out = tf.reshape(lrn1, [-1, weights['out'].get_shape().as_list()[0]])
    out = tf.add(tf.matmul(out, weights['out']), biases['out'])

    # Loss and optimizer
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(out, y))
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    
    # Accuracy
    accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(out, y, 1), tf.float32))
    
    return train_op, x, y, out, loss, accuracy, weights, biases

###############################################################################
### Settings

# Training settings
# Note: Training terminates when the sustained loss is below loss_threshold, or when training has reached max_epochs
max_epochs = 3000
batch_size = 190 # 228 - validation_set_size
validation_set_size = 38
learning_rate = 0.01
dropout = 0.9 # 1.0 = no dropout
loss_threshold = 1e-12
decay_rate = 0.30 # Exponential decay used to calculate sustained loss
use_GPU = True # Use CUDA acceleration

# Weights display settings
show_weights = False
weights_display_step = 10 # in batches
interpolation = None # None to use default (eg. "nearest", "bilinear")
cmap = 'gray' # None to use default (eg. "gray", "inferno")
weights_delay = 0.001
weights_title = 'Weights'

# Loss display settings
show_loss = False
loss_display_step = 10
loss_saved_iterations = float('inf')
loss_starting_iteration = 10
loss_delay = 0.001
loss_title = 'Loss'

# Accuracy display settings
show_accuracy = False
accuracy_display_step = 10
accuracy_saved_iterations = float('inf')
accuracy_starting_iteration = 0
accuracy_delay = 0.001
accuracy_title = 'Accuracy'

# Console output settings (irrelevant if training set size == batch size)
progress_bar_size = 20

# Sound settings
use_sounds = True

##############################################################################
# Data loader and setup

# Load data
print 'Loading data..'

data_source = 'training_data'
data = sio.loadmat(data_source)
good_vals = data["good"]
bad_vals = data["bad"]

def reduce_by_mean(a):
    avg = float(sum(a)) / len(a);
    return map(lambda x: x - avg, a)

# Shuffle data
data = []
for val in good_vals:
    data.append((1, [[[v]] for v in reduce_by_mean(val[900:1000])]))
for val in bad_vals:
    data.append((0, [[[v]] for v in reduce_by_mean(val[900:1000])]))
np.random.shuffle(data)

labels = np.array([int(i[0]) for i in data])
data = np.array([np.array(i[1]) for i in data])

# Create batches
print 'Creating batches..'
assert validation_set_size <= len(data), 'validation_set_size must be smaller than len(data)'

training_data = data[:len(data) - validation_set_size]
training_labels = labels[:len(labels) - validation_set_size]
validation_data = data[len(data) - validation_set_size:]
validation_labels = labels[len(labels) - validation_set_size:]

#validation_data = training_data
#validation_labels = training_labels

print 'Training data: ' + str(len(training_data))
print 'Validation data: ' + str(len(validation_data))

assert float(len(training_data)) / batch_size % 1 == 0, 'batch_size must evenly divide len(training_data)'
#assert float(validation_set_size) / batch_size % 1 == 0, 'batch_size must evenly divide validation_set_size'
num_training_batches = len(training_data) / batch_size
#num_validation_batches = validation_set_size / batch_size
num_validation_batches = 1

training_data_batches = []
training_label_batches = []
validation_data_batches = []
validation_label_batches = []
for i in range(num_training_batches):
    training_data_batches.append(training_data[i*batch_size:(i+1)*batch_size])
    training_label_batches.append(training_labels[i*batch_size:(i+1)*batch_size])
for i in range(num_validation_batches):
    validation_data_batches.append(validation_data[i*batch_size:(i+1)*batch_size])
    validation_label_batches.append(validation_labels[i*batch_size:(i+1)*batch_size])

###############################################################################
# Pyplot setup
plt.ion() # Enable interactive mode
###############################################################################

###############################################################################
# Progress display setup
weights_fig = None
if show_weights:
    weights_fig = plt.figure(weights_title)
    weights_fig.show()
def display_weights(weights_val):
    weights_fig = plt.figure(weights_title)
    weights_fig.clear()
    plot_height = int(weights_val['conv1'].shape[3] ** 0.5)
    plot_width = math.ceil(float(weights_val['conv1'].shape[3]) / plot_height)
    for j in range(weights_val['conv1'].shape[3]):
        ax = weights_fig.add_subplot(plot_height, plot_width, j + 1)
        ax.imshow(np.transpose(weights_val['conv1'][:,:,0,j]), interpolation=interpolation, cmap=cmap)
    plt.pause(weights_delay)
###############################################################################

###############################################################################
# Loss display setup
loss_fig = None
if show_loss:
    loss_fig = plt.figure(loss_title)
    loss_fig.show()
def display_loss(loss_values, iteration):
    loss_fig = plt.figure(loss_title)
    loss_fig.clear()
    plt.plot(list(range(iteration - len(loss_values) + 1, iteration + 1)), loss_values)
    plt.pause(loss_delay)
###############################################################################

###############################################################################
# Accuracy display setup
accuracy_fig = None
if show_loss:
    accuracy_fig = plt.figure(accuracy_title)
    accuracy_fig.show()
def display_accuracy(validation_accuracy_values, max_accuracy_values, iteration):
    accuracy_fig = plt.figure(accuracy_title)
    accuracy_fig.clear()
    plt.plot(list(range(iteration - len(validation_accuracy_values) + 1, iteration + 1)), validation_accuracy_values)
    #plt.plot(list(range(iteration - len(max_accuracy_values) + 1, iteration + 1)), max_accuracy_values)
    #plt.plot([iteration - len(max_accuracy_values) + 1, iteration], [max_accuracy_values[-1]] * 2)
    plt.pause(accuracy_delay)
###############################################################################

###############################################################################
# Sound setup
sounds = Sounds()
if use_sounds:
    sounds.open()
###############################################################################

###############################################################################
# Output control
def update_output(iteration, weights_val, loss_values, validation_accuracy_values, max_accuracy_values, override=False):
    
    # Show/update weights display
    if iteration % weights_display_step == 0 and show_weights or override:
        display_weights(weights_val)

    # Show/update loss display
    if iteration % loss_display_step == 0 and show_loss or override:
        display_loss(loss_values, iteration)

    # Show/update accuracy display
    if iteration % accuracy_display_step == 0 and show_accuracy or override:
        display_accuracy(validation_accuracy_values, max_accuracy_values, iteration)
        
###############################################################################

# Build model and get variable handles
train_op, x, y, out, loss, accuracy, weights, biases = model(learning_rate, dropout)

# Initialize environment
initialize = tf.global_variables_initializer()

# Session config
config = tf.ConfigProto(device_count = {'GPU': 1 if use_GPU == True else 0})

# Run model
done = False
epoch = 0
iteration = 0
sustained_loss = 0.0
loss_values = []
validation_accuracy_values = []
max_accuracy_values = []

max_accuracy = 0.0
max_accuracy_weights = None
max_accuracy_biases = None
    
with tf.Session(config=config) as session:
    session.run(initialize)
    
    print '=========='
    print 'GPU ' + ('enabled' if use_GPU else 'disabled')
    print
    
    # Show weight initialization
    if show_weights:
        weights_val = session.run(weights)
        display_weights(weights_val)
    
    layout = [
        dict(name='Ep.', width=4, align='center'),
        dict(name='Batch', width=2*len(str(num_training_batches))+1, suffix='/'+str(num_training_batches)),
        dict(name='Loss', width=8),
        dict(name='Val Acc', width=6, suffix='%'),
        dict(name='Max Acc', width=6, suffix='%'),
        dict(name='Time', width=progress_bar_size+2, align='center'),
    ]
    table = DynamicConsoleTable(layout)
    table.print_header()
    
    while not done:
        epoch += 1

        # Trains on the data, in batches
        for i in range(num_training_batches):
            iteration += 1
                        
            data_batch = training_data_batches[i]
            label_batch = training_label_batches[i]
            _, loss_val = session.run([train_op, loss], feed_dict={x: data_batch, y: label_batch})
            sustained_loss = decay_rate * sustained_loss + (1.0 - decay_rate) * loss_val
            
            if len(loss_values) == loss_saved_iterations:
                loss_values.pop(0)
            if iteration >= loss_starting_iteration:
                loss_values.append(loss_val)
            
            data_batch = validation_data_batches[iteration % num_validation_batches]
            label_batch = validation_label_batches[iteration % num_validation_batches]
            
            validation_accuracy = 0.0
            for j in range(num_validation_batches):
                data_batch = validation_data_batches[j]
                label_batch = validation_label_batches[j]
                accuracy_val, out_val = session.run([accuracy, out], feed_dict={x: data_batch, y: label_batch})
                
                if epoch >= max_epochs:
                    for j in range(len(label_batch)):
                        print label_batch[j], np.argmax(out_val[j])
                
                validation_accuracy += accuracy_val
            validation_accuracy /= num_validation_batches
            
            if len(validation_accuracy_values) == accuracy_saved_iterations:
                validation_accuracy_values.pop(0)
            if iteration >= accuracy_starting_iteration:
                validation_accuracy_values.append(validation_accuracy)
            
            if validation_accuracy > max_accuracy:
                weights_val, biases_val = session.run([weights, biases])
                max_accuracy = validation_accuracy
                max_accuracy_weights = weights_val
                max_accuracy_biases = biases_val
                if use_sounds:
                    sounds.success()
                    
            if len(max_accuracy_values) == accuracy_saved_iterations:
                max_accuracy_values.pop(0)
            if iteration >= accuracy_starting_iteration:
                max_accuracy_values.append(max_accuracy)
            
            progress = int(math.ceil(progress_bar_size * float((iteration - 1) % num_training_batches) / max(1, num_training_batches - 1)))
            progress_string = '[' + '#' * progress + ' ' * (progress_bar_size - progress) + ']'
            if iteration % num_training_batches == 0:
                progress_string = time.strftime("%I:%M:%S %p", time.localtime())
            table.update(epoch,
                         (iteration - 1) % num_training_batches + 1,
                         sustained_loss,
                         validation_accuracy * 100,
                         max_accuracy * 100,
                         progress_string)
            
            # Termination condition
            if sustained_loss < loss_threshold:
                done = True
                break

            update_output(iteration, weights_val, loss_values, validation_accuracy_values, max_accuracy_values)
        
        table.finalize()
            
        # Termination condition
        if epoch >= max_epochs or sustained_loss < loss_threshold:
            done = True
            update_output(iteration, weights_val, loss_values, validation_accuracy_values, max_accuracy_values, override=True)
            n = 100
            plt.figure('FC weights')
            #plt.plot(weights_val['out'][:,1])
            plt.plot(np.sum([max_accuracy_weights['out'][k*n:(k+1)*n,1] for k in range(num_kernels)], axis = 0))
            #plt.plot(np.sum([weights_val['out'][0:1*n,1], weights_val['out'][1*n:2*n,1], weights_val['out'][2*n:3*n,1], weights_val['out'][3*n:4*n,1], weights_val['out'][4*n:5*n,1], weights_val['out'][5*n:6*n,1]], axis=0))
            plt.show()
            plt.pause(0)
            
