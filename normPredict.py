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

IMAGE_SIZE = 128
NUM_CHANNELS = 1

# Training Parameters
lr_vgg = 0.00005
lr_vgg_fc = 0.001

lr_2_mid = 0.0005
lr_2_end = 0.00005

lr_3_mid = 0.001
lr_3_end = 0.0001 #0.0002

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
    if np.max(_pred_single) > 255.0:
        _pred_single = _pred_single - 255.0/2.0
        _pred_single = _pred_single / np.max(_pred_single) * (255.0/2.0)
        _pred_single = _pred_single + 255.0/2.0
        min_pos = np.unravel_index(np.argmin(_pred_single),_pred_single.shape)
        _pred_single[min_pos[0], min_pos[1], min_pos[2]] = 0;
        _pred_single = np.clip(_pred_single, 0., 255.)
    else:
        max_pos = np.unravel_index(np.argmax(_pred_single),_pred_single.shape)
        _pred_single[max_pos[0], max_pos[1], max_pos[2]] = 255.0;
        _pred_single = np.clip(_pred_single, 0., 255.)

    return _pred_single


# Create the neural network
def conv_net(img, mask, dropout, is_training, reuse):

    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    img = tf.reshape(img, shape=[-1, IMAGE_SIZE, IMAGE_SIZE, 3], name='inputTensor')
    mask = tf.reshape(mask, shape=[-1, IMAGE_SIZE, IMAGE_SIZE, 1], name='inputMask')
    with tf.variable_scope('vgg', reuse = reuse):

        # Convolution Layer with 64 filters and a kernel size of 3
        conv1_1 = tf.layers.conv2d(inputs = img,
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
        vgg_out = tf.image.resize_nearest_neighbor(fc2, size=(32, 32))


    with tf.variable_scope('scale2', reuse = reuse): 

        conv2_img = tf.layers.conv2d(inputs = img,
            filters=96, kernel_size=5, activation=tf.nn.relu, padding='same', name='conv2_img')
        # Max Pooling (down-sampling) with strides of 4 and kernel size of 4
        pool2_img = tf.layers.max_pooling2d(conv2_img, 4, 4)

        conv2_mask = tf.layers.conv2d(inputs = mask,
            filters=48, kernel_size=5, activation=tf.nn.relu, padding='same', name='conv2_mask')
        # Max Pooling (down-sampling) with strides of 4 and kernel size of 4
        pool2_mask = tf.layers.max_pooling2d(conv2_mask, 4, 4)

        # Stack conv2img with vgg_out, aixs = channel
        padding_pool2_img = tf.constant([[0, 0,], [0, 0], [0, 0], [0, 48+64]])
        pool2_img = tf.pad(pool2_img, padding_pool2_img, "CONSTANT")

        padding_pool2_mask = tf.constant([[0, 0,], [0, 0], [0, 0], [96, 64]])
        pool2_mask = tf.pad(pool2_mask, padding_pool2_mask, "CONSTANT")

        padding_vgg_out = tf.constant([[0, 0,], [0, 0], [0, 0], [96+48, 0]])
        vgg_out = tf.pad(vgg_out, padding_vgg_out, "CONSTANT")

        stack2 = pool2_img + pool2_mask + vgg_out

        # Convolution Layer with 96+64 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(inputs = stack2,
            filters=96+48+64, kernel_size=5, activation=tf.nn.relu, padding='same', name='conv1')
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
        scale2_out = tf.image.resize_nearest_neighbor(conv5, size=(128, 128))


    with tf.variable_scope('scale3', reuse = reuse):

        conv3_img = tf.layers.conv2d(inputs = img,
            filters=96, kernel_size=5, activation=tf.nn.relu, padding='same', name='conv3_img')

        conv3_mask = tf.layers.conv2d(inputs = mask,
            filters=48, kernel_size=5, activation=tf.nn.relu, padding='same', name='conv3_mask')

        padding_conv3_img = tf.constant([[0, 0,], [0, 0], [0, 0], [0, 48+3]])
        conv3_img = tf.pad(conv3_img, padding_conv3_img, "CONSTANT")

        padding_conv3_mask = tf.constant([[0, 0,], [0, 0], [0, 0], [96, 3]])
        conv3_mask = tf.pad(conv3_mask, padding_conv3_mask, "CONSTANT")

        padding_scale2_out = tf.constant([[0, 0,], [0, 0], [0, 0], [96+48, 0]])
        scale2_out_padded = tf.pad(scale2_out, padding_scale2_out, "CONSTANT")

        stack3 = conv3_img + conv3_mask + scale2_out_padded

        # Convolution Layer with 96+48+3 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(inputs = stack3,
            filters=96+48+3, kernel_size=5, activation=tf.nn.relu, padding='same', name='conv1')
        # Convolution Layer with 64 filters and a kernel size of 3
        conv2 = tf.layers.conv2d(inputs = conv1,
            filters=64, kernel_size=3, activation=tf.nn.relu, padding='same', name='conv2')
        # Convolution Layer with 64 filters and a kernel size of 3
        conv3 = tf.layers.conv2d(inputs = conv2,
            filters=64, kernel_size=3, activation=tf.nn.relu, padding='same', name='conv3')
        # Convolution Layer with 3 filters and a kernel size of 3
        conv4 = tf.layers.conv2d(inputs = conv3,
            filters=3, kernel_size=3, activation=tf.nn.relu, padding='same', name='conv4')

    out = conv4
    return out

#                                                                         #
# #############################   DATASET   ############################# #
#                                                                         #
num_train_set = 18000
num_self_test_set = 20000-num_train_set
test_batch_size = 20
num_eval_set = 2000
batch_size = 18

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
prediction = conv_net(imageIn, maskIn, Drop_rate, is_training=True, reuse=False)
loss_op = calcloss(prediction, maskIn, gtIn)

# Construct test graph
prediction_slftest = conv_net(imageIn_slftest, maskIn_slftest, dropout=0.0, is_training=False, reuse=True)
loss_test_dotprod_op = caldotprod(prediction_slftest, maskIn_slftest, gtIn_slftest)

# Construct eval graph
prediction_eval = conv_net(imageIn_eval, maskIn_eval, dropout=0.0, is_training=False, reuse=True)

#                                                                         #
# #############################   VAR MAN   ############################# #
#                                                                         #

# Manage vars

# Vgg var
vggConvVar_dict = {getVggStoredName(val):val for val in tf.global_variables() if 'vgg/conv' in val.op.name}
vggConvVar_list = list(vggConvVar_dict.values())

# scale2Var
vggFC_scale2Var_list = [var for var in tf.global_variables() if var not in vggConvVar_list and 'scale3' not in var.op.name]
vggFCVar_list = [var for var in tf.global_variables() if 'vgg/fc' in var.op.name]

scale2EndVar_list = [var for var in tf.global_variables() if 'scale2/conv1' in var.op.name or \
                                                             'scale2/conv5' in var.op.name or \
                                                             '2_img' in var.op.name or \
                                                             '2_mask' in var.op.name]

scale2MidVar_list = [var for var in tf.global_variables() if 'scale2' in var.op.name and var not in scale2EndVar_list]

# scale3Var
scale3Var_list = [var for var in tf.global_variables() if 'scale3' in var.op.name]
scale3EndVar_list = [var for var in scale3Var_list if 'conv1' in var.op.name or \
                                                      'conv4' in var.op.name or \
                                                      'img' in var.op.name or \
                                                      'mask' in var.op.name]

scale3MidVar_list = [var for var in scale3Var_list if var not in scale3EndVar_list]


# Savers
vggConvRestorer = tf.train.Saver(vggConvVar_dict)
scale2Restorer = tf.train.Saver(vggFC_scale2Var_list)
scale3Restorer = tf.train.Saver(scale3Var_list)

# for key, val in vggConvVar_dict.items():
#     print(key, val)
# print("\n\n")
# for t in vggFC_scale2Var_list:
#     print(t.op.name, t.shape)
# print("\n\n")
# for t in vggFCVar_list:
#     print(t.op.name, t.shape)
# print("\n\n")

# for t in scale2MidVar_list:
#     print(t.op.name, t.shape)
# print("\n\n")
# for t in scale2EndVar_list:
#     print(t.op.name, t.shape)
# print("\n\n")

# for t in scale3Var_list:
#     print(t.op.name, t.shape)
# print("\n\n")

# for t in scale3MidVar_list:
#     print(t.op.name, t.shape)
# print("\n\n")
# for t in scale3EndVar_list:
#     print(t.op.name, t.shape)
# print("\n\n")

# exit(0)

#                                                                          #
# ###########################   TRAIN & LOSS   ########################### #
#                                                                          #

num_epochs = 3
display_step = 20
test_step = num_train_set/batch_size # 1 per epoch

# Generate train op

# vgg
# opt_vgg    = tf.train.AdamOptimizer(learning_rate=lr_vgg)
# part 2
# opt_vgg_fc = tf.train.AdamOptimizer(learning_rate=lr_vgg_fc)
# opt_2mid   = tf.train.AdamOptimizer(learning_rate=lr_2_mid)
# opt_2end   = tf.train.AdamOptimizer(learning_rate=lr_2_end)
# part 3
opt_3mid   = tf.train.AdamOptimizer(learning_rate=lr_3_mid)
opt_3end   = tf.train.AdamOptimizer(learning_rate=lr_3_end)

grads = tf.gradients(loss_op, 
    # vggConvVar_list + 
    # vggFCVar_list +
    # scale2MidVar_list +
    # scale2EndVar_list
    scale3MidVar_list +
    scale3EndVar_list
)


# part 2
# grads_vggFC   = grads[0:4]
# grads_s2mid   = grads[4:10]
# grads_s2end   = grads[10:18]

# vgg+2
# grads_vggConv = grads[0:26]
# grads_vggFC   = grads[26:30]
# grads_s2mid   = grads[30:36]
# grads_s2end   = grads[36:44]

# part 3
grads_s3mid   = grads[0:4]
grads_s3end   = grads[4:12]

# train_vggConv = opt_vgg.apply_gradients(zip(grads_vggConv, vggConvVar_list))
# train_vggFC = opt_vgg_fc.apply_gradients(zip(grads_vggFC, vggFCVar_list))
# train_2mid = opt_2mid.apply_gradients(zip(grads_s2mid, scale2MidVar_list))
# train_2end = opt_2end.apply_gradients(zip(grads_s2end, scale2EndVar_list))
train_3mid = opt_3mid.apply_gradients(zip(grads_s3mid, scale3MidVar_list))
train_3end = opt_3end.apply_gradients(zip(grads_s3end, scale3EndVar_list))

train_op = tf.group(
    # train_vggConv, 
    # train_vggFC, 
    # train_2mid, 
    # train_2end 
    train_3mid, 
    train_3end
)

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
    vggConvRestorer.restore(sess, "./savedModel/vgg_BwScale2.ckpt")
    scale2Restorer.restore(sess, "./savedModel/scale2_BwVgg.ckpt")

    # Run
    step = 1
    epoch = 1
    tic = time.time()
    loss_cum = 0
    while True:
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
                dotprod_selftest_cum = []
                for test_batch_i in range(0, int(num_self_test_set/test_batch_size)):
                    dotprod_part, pred_selftest, mask_selftest, gt_selftest = sess.run(loss_test_dotprod_op)
                    dotprod_selftest = np.array(dotprod_part)
                    dotprod_selftest_cum = np.concatenate((dotprod_selftest_cum,dotprod_selftest),axis=0)

                    # SAVE DEBUG IMAGES
                    for id_in_batch in range(0,len(pred_selftest)):
                        # SCALE PREDICTION TO 0-255
                        pred_single = pred_selftest[id_in_batch,:,:,:]

                        scipy.misc.imsave('./test_pred/' + str(test_batch_size * test_batch_i + id_in_batch) + '.png', scalePrediction(pred_single))
                        scipy.misc.imsave('./test_mask/' + str(test_batch_size * test_batch_i + id_in_batch) + '.png', mask_selftest[id_in_batch,:,:])
                        scipy.misc.imsave('./test_gt/' + str(test_batch_size * test_batch_i + id_in_batch) + '.png',   gt_selftest[id_in_batch,:,:,:])

                loss_selftest = -np.mean(dotprod_selftest_cum.astype(np.float32))
                train_loss = loss_cum/test_step
                print("Epoch " + str(epoch) + ", test loss: " + "{:.4f}".format(loss_selftest) + ", train loss: " + "{:.4f}".format(train_loss))
                loss_cum = 0

                toc = time.time() - tic
                print("--------------- Epoch {} done in {:.4f} minutes ---------------".format(epoch, toc/60.))
                tic = time.time()

                epoch = epoch + 1

                # save_pathVgg = vggConvRestorer.save(sess, "./savedModel/vgg_BwScale2.ckpt")
                # print("Saved vgg var at " + save_pathVgg)

                # save_path2 = scale2Restorer.save(sess, "./savedModel/scale2_BwVgg.ckpt")
                # print("Saved scale2 var at " + save_path2)

                save_path3 = scale3Restorer.save(sess, "./savedModel/scale3_B.ckpt")
                print("Saved scale3 var at " + save_path3)

                if loss_selftest > train_loss + 0.02:
                    break

            step = step + 1

        except tf.errors.OutOfRangeError as e:
            break

    

    for eval_batch_i in range(0, int(num_eval_set/test_batch_size)):
        pred_eval = sess.run(prediction_eval)
        # SAVE EVAL IMAGES
        for id_in_batch in range(0,len(pred_eval)):
            scipy.misc.imsave('./submit_out/' + str(test_batch_size * eval_batch_i + id_in_batch) + '.png', scalePrediction(pred_eval[id_in_batch,:,:,:]))

    print("Optimization Finished!")

    sess.close()