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
import time
from PIL import Image

IMAGE_SIZE = 128
NUM_CHANNELS = 1

# Training Parameters
lr = 0.0005
nStack = 2

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

    _mask = tf.decode_raw(_features['mask'], tf.float32)
    _mask = tf.reshape(_mask, [IMAGE_SIZE,IMAGE_SIZE])

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

    _mask = tf.decode_raw(_features['mask'], tf.float32)
    _mask = tf.reshape(_mask, [IMAGE_SIZE,IMAGE_SIZE])

    return _image, _mask

def caldotprod(prediction, mask, groundtruth):
    # normalize prediction
    _prediction = prediction
    maxv = tf.reduce_max(_prediction)
    minv = tf.reduce_min(_prediction)
    _prediction = tf.where(tf.greater_equal(_prediction, tf.zeros_like(_prediction)), 
                            _prediction / maxv, _prediction / (-minv))
    _pred_norm = tf.sqrt(tf.reduce_sum(tf.multiply(_prediction, _prediction), axis=3, keep_dims=True))
    _pred_normalized = tf.div(_prediction, tf.tile(_pred_norm, [1,1,1,3]))
    # normalize groundtruth
    _groundtruth = ((groundtruth / 255.0) - 0.5) * 2.
    _gt_norm = tf.sqrt(tf.reduce_sum(tf.multiply(_groundtruth, _groundtruth), axis=3, keep_dims=True))
    _gt_normalized = tf.div(_groundtruth, tf.tile(_gt_norm, [1,1,1,3]))
    # calculate dot product = cos(theta)
    _dotprod = tf.reduce_sum(tf.multiply(_pred_normalized, _gt_normalized), axis=3)
    # mask object
    _mask = tf.not_equal(mask, tf.zeros_like(mask))
    _dotprod = tf.boolean_mask(_dotprod, _mask)
    # fix nan by setting to -1
    _dotprod = tf.where(tf.is_nan(_dotprod), tf.zeros_like(_dotprod)-1, _dotprod)
    # clip to -1,+1
    _dotprod = tf.clip_by_value(_dotprod, -1., 1.)

    return _dotprod, prediction, mask, groundtruth

# prediction 0-255, groundtruth 0-255
def calcloss(_prediction, _mask, _groundtruth):
    
    _dotprod, _, _, _ = caldotprod(_prediction, _mask, _groundtruth)
    # calculate angles
    _ang     = tf.acos(_dotprod)

    loss = -tf.reduce_mean(_dotprod)
    # loss = tf.reduce_mean(_ang)

    return loss

# vgg/conv1_1_W -- vgg origin
# vgg/conv1_1/kernel:0 -- var
def getVggStoredName(var):
    # get name stored in vgg origin
    if 'kernel' in var.op.name:
        return var.op.name.replace('/kernel','_W')
    elif 'bias' in var.op.name:
        return var.op.name.replace('/bias','_b')
    else:
        print("Error: No kernel or bias")

def scalePrediction(_pred_single):
    maxv = np.amax(_pred_single)
    minv = np.amin(_pred_single)
    _pred_single[_pred_single>0] = _pred_single[_pred_single>0]/maxv*(255./2.)
    _pred_single[_pred_single<0] = _pred_single[_pred_single<0]/(-minv)*(255./2.)
    _pred_single = _pred_single + (255./2.)
    _pred_single = np.clip(_pred_single, 0., 255.)

    return _pred_single

def conv2dBnReLU(inputs, name, num_filter, kernel_size, is_training, reuse):
    with tf.variable_scope(name, reuse=reuse):
        conv = tf.layers.conv2d(inputs = inputs, filters=num_filter, kernel_size=kernel_size, 
                                kernel_regularizer =  tf.contrib.layers.l2_regularizer(scale=0.0005/lr),
                                padding='same', name='conv')
        conv = tf.layers.batch_normalization(conv, scale=False, training=is_training)
        return tf.nn.relu(conv)

def residual(inputs, name, is_training, reuse):
    with tf.variable_scope(name, reuse = reuse):
        conv1 = conv2dBnReLU(inputs=inputs, name='conv1', num_filter=128, kernel_size=1, is_training=is_training, reuse=reuse)
        conv2 = conv2dBnReLU(inputs=conv1,  name='conv2', num_filter=128, kernel_size=3, is_training=is_training, reuse=reuse)
        conv3 = conv2dBnReLU(inputs=conv2,  name='conv3', num_filter=256, kernel_size=1, is_training=is_training, reuse=reuse)
        return conv3+inputs

def hourglass(inputs, name, is_training, reuse):

    with tf.variable_scope(name, reuse = reuse):

        hgL = residual(inputs, 'hgL', is_training=is_training, reuse=reuse)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        low1_in = tf.layers.max_pooling2d(hgL, 2, 2)

        low1 = residual(low1_in, 'low1', is_training=is_training, reuse=reuse)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        low2_in = tf.layers.max_pooling2d(low1, 2, 2)

        low2 = residual(low2_in, 'low2', is_training=is_training, reuse=reuse)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        low3_in = tf.layers.max_pooling2d(low2, 2, 2)

        low3 = residual(low3_in, 'low3', is_training=is_training, reuse=reuse)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        in1_in = tf.layers.max_pooling2d(low3, 2, 2)

        # Inner three blocks
        in1 = residual(in1_in, 'in1', is_training=is_training, reuse=reuse)
        in2 = residual(in1, 'in2', is_training=is_training, reuse=reuse)
        in3 = residual(in2, 'in3', is_training=is_training, reuse=reuse)
        in3 = tf.image.resize_nearest_neighbor(in3, size=(16, 16))

        # branch blocks
        hgL_b = residual(hgL, 'hgL_b', is_training=is_training, reuse=reuse)
        low1_b = residual(low1, 'low1_b', is_training=is_training, reuse=reuse)
        low2_b = residual(low2, 'low2_b', is_training=is_training, reuse=reuse)
        low3_b = residual(low3, 'low3_b', is_training=is_training, reuse=reuse)

        # remaining network
        
        up1_in = low3_b + in3
        up1 = residual(up1_in, 'up1', is_training=is_training, reuse=reuse)
        up1 = tf.image.resize_nearest_neighbor(up1, size=(32, 32))

        up2_in = low2_b + up1
        up2 = residual(up2_in, 'up2', is_training=is_training, reuse=reuse)
        up2 = tf.image.resize_nearest_neighbor(up2, size=(64, 64))

        up3_in = low1_b + up2
        up3 = residual(up3_in, 'up3', is_training=is_training, reuse=reuse)
        up3 = tf.image.resize_nearest_neighbor(up3, size=(128, 128))

        hgR_in = hgL_b + up3
        hgR = residual(hgR_in, 'hgR', is_training=is_training, reuse=reuse)

        return hgR

def hourglass_int(inputs, name, is_training, reuse):

    with tf.variable_scope(name, reuse=reuse):

        hg_Out = hourglass(inputs, '_hourglass', is_training=is_training, reuse=reuse)
        res1 = residual(hg_Out, 'res1', is_training=is_training, reuse=reuse)
        res2 = residual(res1, 'res2', is_training=is_training, reuse=reuse)

        # intermediate prediction
        int_pred = tf.layers.conv2d(inputs = res1, filters=3, kernel_size=1,
                                    kernel_regularizer =  tf.contrib.layers.l2_regularizer(scale=0.0005/lr),
                                    padding='same', name='int_pred')

        res3 = tf.layers.conv2d(inputs = int_pred, filters=256, kernel_size=1, activation=tf.nn.relu,
                                kernel_regularizer =  tf.contrib.layers.l2_regularizer(scale=0.0005/lr),
                                padding='same', name='res3')

        return res3 + res2 + inputs, int_pred

def preprocess(inputs, is_training, reuse):

    with tf.variable_scope('pre', reuse=reuse):

        conv1 = conv2dBnReLU(inputs=inputs, name='conv1', num_filter=256, kernel_size=3, is_training=is_training, reuse=reuse)
        res1 = residual(conv1, 'res1', is_training=is_training, reuse=reuse)
        res2 = residual(res1,  'res2', is_training=is_training, reuse=reuse)
        return res2

# Create the neural network
def conv_net(img, mask, dropout, is_training, reuse):

    img = tf.reshape(img, shape=[-1, IMAGE_SIZE, IMAGE_SIZE, 3], name='inputTensor')
    mask = tf.reshape(mask, shape=[-1, IMAGE_SIZE, IMAGE_SIZE, 1], name='inputMask')
    
    padding_img = tf.constant([[0, 0,], [0, 0], [0, 0], [0, 1]])
    img = tf.pad(img, padding_img, "CONSTANT")

    padding_mask = tf.constant([[0, 0,], [0, 0], [0, 0], [3, 0]])
    mask = tf.pad(mask, padding_mask, "CONSTANT")

    imgWmask = img + mask

    ret = [None] * nStack # int1 int2 .. Final
    hgout = [None] * nStack

    with tf.variable_scope('ConvNet', reuse=reuse):

        preprocessed = preprocess(imgWmask, is_training=is_training, reuse=reuse)

        hgout[0], ret[0] = hourglass_int(preprocessed, 'hourglass.0', is_training=is_training, reuse=reuse)

        for i in range(1, nStack-1):
            hgout[i], ret[i] = hourglass_int(hgout[i-1], 'hourglass.'+str(i), is_training=is_training, reuse=reuse)

        hgout[-1] = hourglass(hgout[nStack-2], 'hourglass.out', is_training=is_training, reuse=reuse)
        res1 = residual(hgout[-1], 'res1', is_training=is_training, reuse=reuse)
        res2 = residual(res1, 'res2',  is_training=is_training, reuse=reuse)

        fc1 = tf.layers.conv2d(inputs = res2, filters=256, kernel_size=1, activation=tf.nn.relu,
                               kernel_regularizer =  tf.contrib.layers.l2_regularizer(scale=0.0005/lr),
                               padding='same', name='fc1')
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)
        ret[-1] = tf.layers.conv2d(inputs = fc1, filters=3, kernel_size=1,
                               kernel_regularizer =  tf.contrib.layers.l2_regularizer(scale=0.0005/lr),
                               padding='same', name='predfc')


        return ret
#                                                                         #
# #############################   DATASET   ############################# #
#                                                                         #
num_train_set = 19800
num_self_test_set = 20000-num_train_set
test_batch_size = 5
num_eval_set = 2000
batch_size = 5

trainingSetFiles = ['./trainTFRecords/' + str(I) +'.tfrecords' for I in range(0,num_train_set)]
slfTestSetFiles = ['./trainTFRecords/' + str(I) +'.tfrecords' for I in range(num_train_set,20000)]
evalSetFiles = ['./testTFRecords/' + str(I) +'.tfrecords' for I in range(0,2000)]

# construct dataset
# training set
trainSet = tf.data.TFRecordDataset(trainSetName).map(_parser).shuffle(buffer_size=3600).repeat().batch(batch_size)
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
predictions = conv_net(imageIn, maskIn, Drop_rate, is_training=True, reuse=False)
losses = [None] * nStack
weights = tf.pow(tf.constant(10.), tf.constant(list(range(nStack)),dtype=tf.float32)+1.)
weights = weights / tf.reduce_sum(weights)
for i in range(nStack):
    losses[i]  = calcloss(predictions[i], maskIn, gtIn) * weights[i]

loss_op = tf.reduce_sum(losses)

# Construct test graph
test_predictions = conv_net(imageIn_slftest, maskIn_slftest, dropout=0.0, is_training=False, reuse=True)
test_dotprods_op = caldotprod(test_predictions[-1], maskIn_slftest, gtIn_slftest)

# Construct eval graph
eval_predictions = conv_net(imageIn_eval, maskIn_eval, dropout=0.0, is_training=False, reuse=True)
prediction_eval = eval_predictions[-1]

# for name in sorted([var.op.name for var in tf.global_variables()]):
#     print(name)

# exit(0)
#                                                                         #
# #############################   VAR MAN   ############################# #
#                                                                         #

# Manage vars
saveVar = tf.global_variables()

# Savers
restorer = tf.train.Saver(saveVar)

#                                                                          #
# ###########################   TRAIN & LOSS   ########################### #
#                                                                          #

num_epochs = 20
display_step = 100
test_step = num_train_set/batch_size # 1 per epoch

# Generate train op

optimizer  = tf.train.AdamOptimizer(learning_rate=lr)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(loss_op)

# Initialize the variables (i.e. assign their default value)
init_val = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # writer = tf.summary.FileWriter("./graph", sess.graph)

    # Run the initializer
    sess.run(tf.group(init_val, it_train.initializer, it_selftest.initializer, it_eval.initializer), \
            feed_dict={
                        trainSetName:   trainingSetFiles,
                        slfTestSetName: slfTestSetFiles,
                        evalSetName:    evalSetFiles
            }
    )
    # restorer.restore(sess, "./savedModel/hg2_1.ckpt")

    # Run
    step = 1
    epoch = 1
    tic = time.time()
    loss_cum = 0
    loss_selftest_prev = 0
    min_loss = 0
    inferenceOnly = True
    while not inferenceOnly:
    # for i in range(0,10):
        try:
            # Run optimization op (backprop)
            loss, _ = sess.run([loss_op, train_op], feed_dict={Drop_rate: 0.5})
            loss_cum = loss_cum + loss
            # Display training loss
            if step % display_step == 0 or step == 1:
                print("Epoch " + str(epoch) + ", " + \
                      "Step " + str(step) + ", Mini batch Loss= " + "{:.4f}".format(loss))
            # Display self-test loss
            if step % test_step == 0:
                dotprod_selftest_join = []
                for test_batch_i in range(0, int(num_self_test_set/test_batch_size)):
                    dotprod_part, pred_selftest, mask_selftest, gt_selftest = sess.run(test_dotprods_op)
                    dotprod_selftest = np.array(dotprod_part)
                    dotprod_selftest_join = np.concatenate((dotprod_selftest_join,dotprod_selftest),axis=0)

                    # SAVE DEBUG IMAGES
                    for id_in_batch in range(0,len(pred_selftest)):
                        # SCALE PREDICTION TO 0-255
                        pred_single = pred_selftest[id_in_batch,:,:,:]

                        Image.fromarray(scalePrediction(pred_single).astype(np.uint8)) \
                            .save('./test_pred/' + str(test_batch_size * test_batch_i + id_in_batch) + '.png')
                        Image.fromarray(mask_selftest[id_in_batch,:,:].astype(np.uint8)) \
                            .save('./test_mask/' + str(test_batch_size * test_batch_i + id_in_batch) + '.png')
                        Image.fromarray(gt_selftest[id_in_batch,:,:,:].astype(np.uint8)) \
                            .save('./test_gt/' + str(test_batch_size * test_batch_i + id_in_batch) + '.png')

                loss_selftest = - np.mean(dotprod_selftest_join.astype(np.float32))
                train_loss = loss_cum/test_step
                print("Epoch " + str(epoch) + ", test loss: " + "{:.4f}".format(loss_selftest) + ", train loss: " + "{:.4f}".format(train_loss))
                loss_cum = 0

                toc = time.time() - tic
                print("--------------- Epoch {} done in {:.4f} minutes ---------------".format(epoch, toc/60.))
                tic = time.time()

                epoch = epoch + 1

                if loss_selftest < min_loss:
                    save_path = restorer.save(sess, "./savedModel/hg2_1.ckpt")
                    print("Saved var at " + save_path)
                    min_loss = loss_selftest
                else:
                    break

                if abs(loss_selftest - loss_selftest_prev) <= 0.0001:
                    break

                loss_selftest_prev = loss_selftest

            step = step + 1

        except tf.errors.OutOfRangeError as e:
            break

    for eval_batch_i in range(0, int(num_eval_set/test_batch_size)):
        pred_eval = sess.run(prediction_eval)
        # SAVE EVAL IMAGES
        for id_in_batch in range(0,len(pred_eval)):
            Image.fromarray(scalePrediction(pred_eval[id_in_batch,:,:,:]).astype(np.uint8)) \
                            .save('./submit_out/' + str(test_batch_size * eval_batch_i + id_in_batch) + '.png')

    print("Optimization Finished!")

    sess.close()