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
learning_rate = 0.001
num_epochs = 3
display_step = 20
test_step = 500
batch_size = 18

# tf Graph input
trainSetName = tf.placeholder(tf.string, shape=[None])
slfTestSetName = tf.placeholder(tf.string,  shape=[None])
evalSetName = tf.placeholder(tf.string, shape=[None])
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

def _parser_eval(proto):
    # Conver tfrecord to tensors
    _features = tf.parse_single_example(
        proto,
        features = {
        'image': tf.FixedLenFeature((),tf.string),
        'mask': tf.FixedLenFeature((),tf.string)
    })

    _image = tf.decode_raw(_features['image'], tf.float32)
    _image = tf.reshape(_image, [IMAGE_SIZE,IMAGE_SIZE,3])

    _mask = tf.decode_raw(_features['mask'], tf.int64)
    _mask = tf.cast(tf.reshape(_mask, [IMAGE_SIZE,IMAGE_SIZE]), tf.bool)

    return _image, _mask

def caldotprod(prediction, _mask, groundtruth):
    # normalize prediction
    _prediction = ((prediction / 255.0) - 0.5) * 2.
    _pred_norm = tf.sqrt(tf.reduce_sum(tf.multiply(_prediction, _prediction), axis=3, keep_dims=True))
    _pred_normalized = tf.div(_prediction, tf.tile(_pred_norm, [1,1,1,3]))
    # normalize groundtruth
    _groundtruth = ((groundtruth / 255.0) - 0.5) * 2.
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

    return _dotprod, prediction, groundtruth

# prediction 0-255, groundtruth 0-255
def calcloss(_prediction, _mask, _groundtruth):
    
    _dotprod, _, _ = caldotprod(_prediction, _mask, _groundtruth)
    # calculate angles
    _ang     = tf.acos(_dotprod)

    loss = -tf.reduce_mean(_dotprod)
    # loss = tf.reduce_mean(_ang)

    return loss

# vgg/conv1_1_W -- vgg origin
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
def conv_net(x, dropout, is_training, reuse):

    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, 128, 128, 3], name='inputTensor')
    with tf.variable_scope('vgg', reuse = reuse):

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

    with tf.variable_scope('scale2', reuse = reuse): 

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

#                                                                         #
# #############################   DATASET   ############################# #
#                                                                         #
num_train_set = 18000
num_self_test_set = 20000-num_train_set
test_batch_size = 100
num_eval_set = 2000

trainingSetFiles = ['./trainTFRecords/' + str(I) +'.tfrecords' for I in range(0,num_train_set)]
slfTestSetFiles = ['./trainTFRecords/' + str(I) +'.tfrecords' for I in range(num_train_set,20000)]
evalSetFiles = ['./testTFRecords/' + str(I) +'.tfrecords' for I in range(0,2000)]

# construct dataset
# training set
trainSet = tf.data.TFRecordDataset(trainSetName).map(_parser).shuffle(buffer_size=1800).repeat().batch(batch_size)
it_train = trainSet.make_initializable_iterator()

# self-testing set
selftestSet = tf.data.TFRecordDataset(slfTestSetName).map(_parser).repeat().batch(test_batch_size)
it_selftest = selftestSet.make_initializable_iterator()

# test set (to upload)
evalSet = tf.data.TFRecordDataset(evalSetName).map(_parser_eval).batch(test_batch_size)
it_eval = evalSet.make_initializable_iterator()

#                                                                         #
# ##############################   BATCH   ############################## #
#                                                                         #

# get training/testing data
# N*128*128*3-(0,1), N*128*128-(0/1), N*128*128*3-(-1,1)
imageIn, maskIn, gtIn = it_train.get_next()
imageIn_slftest, maskIn_slftest, gtIn_slftest = it_selftest.get_next()
imageIn_eval, maskIn_eval = it_eval.get_next() 

#                                                                         #
# ###########################   PRED & LOSS   ########################### #
#                                                                         #

# Construct model
prediction = conv_net(imageIn, Drop_rate, is_training=True, reuse=False)
loss_op = calcloss(prediction, maskIn, gtIn)

# Construct test graph
prediction_slftest = conv_net(imageIn_slftest, dropout=0.0, is_training=False, reuse=True)
loss_test_dotprod_op = caldotprod(prediction_slftest, maskIn_slftest, gtIn_slftest)

# Construct eval graph
prediction_eval = conv_net(imageIn_eval, dropout=0.0, is_training=False, reuse=True)

#                                                                         #
# #############################   VAR MAN   ############################# #
#                                                                         #

# Manage vars

# Vgg var
vggVar_dict = {getVggStoredName(val):val for val in tf.global_variables() \
            if 'vgg/conv' in val.name and 'Adam' not in val.name}
vggRestorer = tf.train.Saver(vggVar_dict)
vggVar_list = vggVar_dict.values()

# scale2Var
scale2Var_list = [var for var in tf.global_variables() if var not in vggVar_list]
scale2Restorer = tf.train.Saver(scale2Var_list)

for key, val in vggVar_dict.items():
    print(key, val)
for t in scale2Var_list:
    print(t.name, t.shape)
#exit(0)

#                                                                          #
# ###########################   TRAIN & LOSS   ########################### #
#                                                                          #

# Generate train op
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op, var_list=scale2Var_list)
# train_op = optimizer.minimize(loss_op)

# Initialize the variables (i.e. assign their default value)
init_val = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    writer = tf.summary.FileWriter("./graph", sess.graph)

    # Run the initializer
    sess.run(tf.group(init_val, it_train.initializer, it_selftest.initializer, it_eval.initializer), \
            feed_dict={
                        trainSetName:   trainingSetFiles,
                        slfTestSetName: slfTestSetFiles,
                        evalSetName:    evalSetFiles
            }
    )
    vggRestorer.restore(sess, "./savedModel/vgg_original.ckpt")
    # scale2Restorer.restore(sess, "./savedModel/scale2.ckpt")

    # image_debug, mask_debug, gt_debug = sess.run([imageIn, maskIn, gtIn])

    # image_debug = np.array(image_debug)
    # mask_debug = np.array(mask_debug).astype(np.int32)*255
    # gt_debug = np.array(gt_debug)
    
    # for i in range(0,len(image_debug)):
    #     scipy.misc.imsave('./tmp/image_debug'+str(i)+'.png',image_debug[i,:,:,:])
    #     scipy.misc.imsave('./tmp/mask_debug'+str(i)+'.png',mask_debug[i,:,:])
    #     scipy.misc.imsave('./tmp/gt_debug'+str(i)+'.png',gt_debug[i,:,:,:])
    # exit(0)

    # Run
    step = 1
    epoch = 1
    while True:
    # for i in range(0,10):
        try:
            # Run optimization op (backprop)
            loss, _ = sess.run([loss_op, train_op], feed_dict={Drop_rate: 0.5})
            # Display training loss
            if step % display_step == 0 or step == 1:
                print("Epoch " + str(epoch) + ", " + \
                      "Step " + str(step) + ", Mini batch Loss= " + "{:.4f}".format(loss))
            # Display self-test loss
            if step % test_step == 0:
                dotprod_selftest_cum = []
                for test_batch_i in range(0, int(num_self_test_set/test_batch_size)):
                    dotprod_part, pred_debug, gt_debug = sess.run(loss_test_dotprod_op)
                    dotprod_selftest = np.array(dotprod_part)
                    dotprod_selftest_cum = np.concatenate((dotprod_selftest_cum,dotprod_selftest),axis=0)
                    
                    # SAVE DEBUG IMAGES
                    for id_in_batch in range(0,len(pred_debug)):
                        scipy.misc.imsave('./debug_out/test_pred' + str(test_batch_size * test_batch_i + id_in_batch) + '.png', pred_debug[id_in_batch,:,:,:])
                        scipy.misc.imsave('./debug_out/test_gt' + str(test_batch_size * test_batch_i + id_in_batch) + '.png',   gt_debug[id_in_batch,:,:,:])

                loss_selftest = -np.mean(dotprod_selftest_cum.astype(np.float32))
                print("Epoch " + str(epoch) + ", test loss: " + "{:.4f}".format(loss_selftest))

            if step*batch_size % num_train_set == 0:
                epoch = epoch + 1

            if epoch > num_epochs:
                break

            step = step + 1

        except tf.errors.OutOfRangeError as e:
            break

    save_path = scale2Restorer.save(sess, "./savedModel/scale2.ckpt")
    print("Saved scale2 var at " + save_path)

    for eval_batch_i in range(0, int(num_eval_set/test_batch_size)):
        pred_eval = sess.run(prediction_eval)
        # SAVE EVAL IMAGES
        for id_in_batch in range(0,len(pred_eval)):
            scipy.misc.imsave('./submit_out/' + str(test_batch_size * eval_batch_i + id_in_batch) + '.png', pred_eval[id_in_batch,:,:,:])

    print("Optimization Finished!")

    sess.close()