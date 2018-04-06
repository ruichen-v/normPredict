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
import imageio
import os
import matplotlib.pyplot as plt
import scipy.misc

IMAGE_SIZE = 128
NUM_CHANNELS = 1

# Training Parameters
learning_rate = 0.01
num_epochs = 2
display_step = 10
batch_size = 32

# tf Graph input
datasetName = tf.placeholder(tf.string, shape=[None])
Drop_rate = tf.placeholder(tf.float32) # dropout (keep probability)
Is_training = tf.placeholder(tf.bool)


def _parser(proto):
    # Conver tfrecord to tensors
    _features = tf.parse_single_example(
        proto,
        features = {
        'image': tf.FixedLenFeature((),tf.string),
        'mask': tf.FixedLenFeature((),tf.string),
        'gt': tf.FixedLenFeature((),tf.string)
    })

    _image = tf.decode_raw(_features['image'], tf.float32)
    _image = tf.reshape(_image, [IMAGE_SIZE,IMAGE_SIZE,3])

    _mask = tf.decode_raw(_features['mask'], tf.int64)
    _mask = tf.cast(tf.reshape(_mask, [IMAGE_SIZE,IMAGE_SIZE]), tf.bool)

    _gt = tf.decode_raw(_features['gt'], tf.float32)
    _gt = tf.reshape(_gt, [IMAGE_SIZE,IMAGE_SIZE,3])

    return _image, _mask, _gt


# prediction 0-255, groundtruth 0-255
def calcloss(_prediction, _mask, _groundtruth):
    # normalize prediction
    _prediction = ((_prediction / 255.0) - 0.5) * 2.
    _pred_norm = tf.sqrt(tf.reduce_sum(tf.multiply(_prediction, _prediction), axis=3, keep_dims=True))
    _pred_normalized = tf.div(_prediction, tf.tile(_pred_norm, [1,1,1,3]))
    # normalize groundtruth
    _groundtruth = ((_groundtruth / 255.0) - 0.5) * 2.
    _gt_norm = tf.sqrt(tf.reduce_sum(tf.multiply(_groundtruth, _groundtruth), axis=3, keep_dims=True))
    _gt_normalized = tf.div(_groundtruth, tf.tile(_gt_norm, [1,1,1,3]))
    # calculate dot product = cos(theta)
    _dotprod = tf.reduce_sum(tf.multiply(_pred_normalized, _gt_normalized), axis=3)
    # mask object
    _dotprod = tf.boolean_mask(_dotprod, _mask)
    # fix nan by setting to -1
    _dotprod = tf.where(tf.is_nan(_dotprod), tf.zeros_like(_dotprod)-1, _dotprod)
    # clip to -1,+1
    _dotprod = tf.clip_by_value(_dotprod, -1., 1.)
    # calculate angles
    _ang     = tf.acos(_dotprod)

    loss = -tf.reduce_sum(_dotprod) / batch_size
    # loss = tf.reduce_mean(_ang)

    return loss

# vgg/conv1_1_W:0 -- vgg origin
# vgg/conv1_1/kernel:0 -- var
def getVggStoredName(var):
    # get name stored in vgg origin
    if 'kernel' in var.name:
        return var.name.replace('/kernel:0','_W')
    elif 'bias' in var.name:
        return var.name.replace('/bias:0','_b')
    else:
        print("Error: No kernel or bias")

# Create the neural network
def conv_net(x, dropout, is_training):
    # Define a scope for reusing the variables
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, 128, 128, 3], name='inputTensor')
    with tf.variable_scope('vgg'):

        # Convolution Layer with 64 filters and a kernel size of 3
        conv1_1 = tf.layers.conv2d(inputs = x,
            filters=64, kernel_size=3, activation=tf.nn.relu, padding='same', name='conv1_1')
        conv1_2 = tf.layers.conv2d(inputs = conv1_1,
            filters=64, kernel_size=3, activation=tf.nn.relu, padding='same', name='conv1_2')
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        pool1 = tf.layers.max_pooling2d(conv1_2, 2, 2)

        # Convolution Layer with 32 filters and a kernel size of 3
        conv2_1 = tf.layers.conv2d(inputs = pool1,
            filters=128, kernel_size=3, activation=tf.nn.relu, padding='same', name='conv2_1')
        conv2_2 = tf.layers.conv2d(inputs = conv2_1,
            filters=128, kernel_size=3, activation=tf.nn.relu, padding='same', name='conv2_2')
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        pool2 = tf.layers.max_pooling2d(conv2_2, 2, 2)

        # Convolution Layer with 256 filters and a kernel size of 3
        conv3_1 = tf.layers.conv2d(inputs = pool2,
            filters=256, kernel_size=3, activation=tf.nn.relu, padding='same', name='conv3_1')
        conv3_2 = tf.layers.conv2d(inputs = conv3_1,
            filters=256, kernel_size=3, activation=tf.nn.relu, padding='same', name='conv3_2')
        conv3_3 = tf.layers.conv2d(inputs = conv3_2,
            filters=256, kernel_size=3, activation=tf.nn.relu, padding='same', name='conv3_3')
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        pool3 = tf.layers.max_pooling2d(conv3_3, 2, 2)

        # Convolution Layer with 512 filters and a kernel size of 3
        conv4_1 = tf.layers.conv2d(inputs = pool3,
            filters=512, kernel_size=3, activation=tf.nn.relu, padding='same', name='conv4_1')
        conv4_2 = tf.layers.conv2d(inputs = conv4_1,
            filters=512, kernel_size=3, activation=tf.nn.relu, padding='same', name='conv4_2')
        conv4_3 = tf.layers.conv2d(inputs = conv4_2,
            filters=512, kernel_size=3, activation=tf.nn.relu, padding='same', name='conv4_3')
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        pool4 = tf.layers.max_pooling2d(conv4_3, 2, 2)

        # Convolution Layer with 512 filters and a kernel size of 3
        conv5_1 = tf.layers.conv2d(inputs = pool4,
            filters=512, kernel_size=3, activation=tf.nn.relu, padding='same', name='conv5_1')
        conv5_2 = tf.layers.conv2d(inputs = conv5_1,
            filters=512, kernel_size=3, activation=tf.nn.relu, padding='same', name='conv5_2')
        conv5_3 = tf.layers.conv2d(inputs = conv5_2,
            filters=512, kernel_size=3, activation=tf.nn.relu, padding='same', name='conv5_3')
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        pool5 = tf.layers.max_pooling2d(conv5_3, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(pool5)

        # Fully connected layer (in tf contrib folder for now)
        fc1 = tf.layers.dense(fc1, 4096, name='fc1')
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training) # drop = 0.5
        fc1 = tf.nn.relu(fc1)

        # Fully connected layer (in tf contrib folder for now)
        fc2 = tf.layers.dense(fc1, 8*8*64, name='fc2')
        fc2 = tf.reshape(fc2, [-1, 8, 8, 64])
        fc2 = tf.nn.relu(fc2)

        # Upsample
        vgg_out = tf.image.resize_bilinear(fc2, size=(32, 32))

    with tf.variable_scope('scale2'): 

        conv2_img = tf.layers.conv2d(inputs = x,
            filters=96, kernel_size=5, activation=tf.nn.relu, padding='same', name='conv2_img')
        # Max Pooling (down-sampling) with strides of 4 and kernel size of 4
        pool2_img = tf.layers.max_pooling2d(conv2_img, 4, 4)

        # Stack conv2img with vgg_out, aixs = channel
        padding_pool2_img = tf.constant([[0, 0,], [0, 0], [0, 0], [0, 64]])
        pool2_img = tf.pad(pool2_img, padding_pool2_img, "CONSTANT")

        padding_vgg_out = tf.constant([[0, 0,], [0, 0], [0, 0], [96, 0]])
        vgg_out = tf.pad(vgg_out, padding_vgg_out, "CONSTANT")

        stack2 = tf.add(pool2_img, vgg_out)
        # stack2 = tf.concat([pool2_img, vgg_out], axis=3) this does not work !!
        # print(stack2.shape)

        # Convolution Layer with 96+64 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(inputs = stack2,
            filters=96+64, kernel_size=5, activation=tf.nn.relu, padding='same', name='conv1')
        # Convolution Layer with 64 filters and a kernel size of 3
        conv2 = tf.layers.conv2d(inputs = conv1,
            filters=64, kernel_size=3, activation=tf.nn.relu, padding='same', name='conv2')
        # Convolution Layer with 64 filters and a kernel size of 3
        conv3 = tf.layers.conv2d(inputs = conv2,
            filters=64, kernel_size=3, activation=tf.nn.relu, padding='same', name='conv3')
        # Convolution Layer with 64 filters and a kernel size of 3
        conv4 = tf.layers.conv2d(inputs = conv3,
            filters=64, kernel_size=3, activation=tf.nn.relu, padding='same', name='conv4')
        # Convolution Layer with 3 filters and a kernel size of 3
        conv5 = tf.layers.conv2d(inputs = conv4,
            filters=3, kernel_size=3, activation=tf.nn.relu, padding='same', name='conv5')
        
        # Upsample
        out = tf.image.resize_bilinear(conv5, size=(128, 128))

    ###### scale 3
    return out

trainingSetNames = ['./trainTFRecords/' + str(I) +'.tfrecords' for I in range(0,20000)]
# construct dataset
dataset = tf.data.TFRecordDataset(datasetName).map(_parser).shuffle(buffer_size=4096).batch(batch_size)

it = dataset.make_initializable_iterator()

# N*128*128*3-(0,1), N*128*128-(0/1), N*128*128*3-(-1,1)
imageIn, maskIn, gtIn = it.get_next()
print(imageIn.shape, maskIn.shape, gtIn.shape)

# Construct model
prediction = conv_net(imageIn, Drop_rate, Is_training)
# Define loss and optimizer
loss_op = calcloss(prediction, maskIn, gtIn)

# Manage vars
vggVar_dict = {getVggStoredName(val):val for val in tf.global_variables() \
            if 'vgg/conv' in val.name and 'Adam' not in val.name}
vggRestorer = tf.train.Saver(vggVar_dict)
vggVar_list = vggVar_dict.values()
trainVar_list = [var for var in tf.global_variables() if var not in vggVar_list]

# for key, val in vggVar_dict.items():
#     print(key, val)
# for t in trainVar_list:
#     print(t.name, t.shape)
# exit(0)

# Generate train op
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op, var_list=trainVar_list)

# Initialize the variables (i.e. assign their default value)
init_val = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run([init_val, it.initializer], feed_dict={datasetName: trainingSetNames})
    vggRestorer.restore(sess, "./savedModel/vgg_original.ckpt")

    # Run
    step = 1
    while True:
    # for i in range(0,10):
        try:
            # Run optimization op (backprop)
            loss, _ = sess.run([loss_op, train_op], feed_dict={Drop_rate: 0.5, Is_training: True})
            if step % display_step == 0 or step == 1:
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss))
                # print(prediction_debug.shape, gtin_debug.shape, maskIn_debug.shape, dotprod_debug)
                # scipy.misc.imsave('prediction_debug.png', prediction_debug[0,:,:,:])
                # scipy.misc.imsave('gt.png', gtin_debug[0,:,:,:])
                # scipy.misc.imsave('mask.png', np.array(maskIn_debug[0,:,:]).astype(np.uint8))
            step = step + 1
        except tf.errors.OutOfRangeError as e:
            break

    print("Optimization Finished!")
    # Create saver
    # saver = tf.train.Saver()
    # save_path = saver.save(sess, "./models/model1.ckpt")
    # saver.restore(sess, "./models/model1.ckpt")

    # sess.run(test_init_op)
    # # Calculate accuracy for 256 MNIST test images
    # print("Testing Accuracy:", \
    #     sess.run(accuracy, feed_dict={Drop_rate: 0.5, Is_training: False}))
    sess.close()