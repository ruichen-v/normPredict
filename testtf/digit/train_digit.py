""" Convolutional Neural Network.
Build and train a convolutional neural network with TensorFlow.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import gzip

IMAGE_SIZE = 28
NUM_CHANNELS = 1

# Training Parameters
learning_rate = 0.0001
num_epochs = 2
display_step = 10

# Network Parameters
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])
Drop_rate = tf.placeholder(tf.float32) # dropout (keep probability)
Is_training = tf.placeholder(tf.bool)

def _extract_img(filename, numOfImages):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    print('Extracting...', filename)
    with gzip.open(filename, 'rb') as _bytestream:
        _bytestream.read(16)
        _buf = _bytestream.read(IMAGE_SIZE * IMAGE_SIZE * numOfImages)
        _data = np.frombuffer(_buf, dtype=np.uint8).astype(np.float32)
        _data = _data.reshape(numOfImages, IMAGE_SIZE * IMAGE_SIZE)
    return _data


def _extract_label(filename, numOfImages):
    """Extract the labels into a vector of int64 label IDs."""
    print('Extracting...', filename)
    with gzip.open(filename, 'rb') as _bytestream:
        _bytestream.read(8)
        _buf = _bytestream.read(1 * numOfImages)
        _labels = np.frombuffer(_buf, dtype=np.uint8).astype(np.int64)
    return _labels

def loadDataAndParse(imageFileName, labelFileName, numOfImages):
    """Load images and lables according to names"""
    _images = _extract_img(imageFileName, numOfImages)
    _labels = _extract_label(labelFileName, numOfImages)
    # convert to one_hot
    _labels = np.eye(num_classes)[_labels]
    return _images, _labels

# Create the neural network
def conv_net(x, n_classes, dropout, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=tf.AUTO_REUSE):

        # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        x = tf.reshape(x, shape=[-1, 28, 28, 1], name='inputTensor')

        # Convolution Layer with 32 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu, name='conv1')
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        # Convolution Layer with 64 filters and a kernel size of 3
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu, name='conv2')
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv2)

        # Fully connected layer (in tf contrib folder for now)
        fc1 = tf.layers.dense(fc1, 1024, name='fc1')
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes, name='predict')

    return out

TRAIN_IMG_filename = "train-images-idx3-ubyte.gz"
TRAIN_LBL_filename = "train-labels-idx1-ubyte.gz"
TEST_IMG_filename = "t10k-images-idx3-ubyte.gz"
TEST_LBL_filename = "t10k-labels-idx1-ubyte.gz"

# load data
train_img, train_labels = loadDataAndParse(TRAIN_IMG_filename, TRAIN_LBL_filename, 60000)
test_img,  test_labels  = loadDataAndParse(TEST_IMG_filename, TEST_LBL_filename, 10000)
# construct dataset
train_set = tf.data.Dataset.from_tensor_slices((train_img, train_labels))\
            .shuffle(buffer_size=10000).batch(128)
test_set = tf.data.Dataset.from_tensor_slices((test_img, test_labels))\
            .shuffle(buffer_size=1000).batch(256)

it = tf.data.Iterator.from_structure(train_set.output_types, train_set.output_shapes)
features, labels = it.get_next()

training_init_op = it.make_initializer(train_set)
test_init_op = it.make_initializer(test_set)

# Construct model
logits = conv_net(features, num_classes, Drop_rate, Is_training)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=labels))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)


# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init_val = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run([init_val, training_init_op])
    step = 1
    # Run
    while True:
        try:
            # Run optimization op (backprop)
            _, loss, acc = sess.run([train_op, loss_op, accuracy], feed_dict={Drop_rate: 0.0, Is_training: True})
            if step % display_step == 0 or step == 1:
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))
            step = step + 1
        except tf.errors.OutOfRangeError as e:
            break
        

    print("Optimization Finished!")
    # for t in tf.global_variables():
    #     print(t)

    sess.run(test_init_op)
    # Calculate accuracy for 256 MNIST test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={Drop_rate: 0.0, Is_training: False}))