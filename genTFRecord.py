import tensorflow as tf
from random import shuffle
import glob
import os
import numpy as np
import imageio
import sys

pathTrainColor = './img/train/color'
pathTrainMask  = './img/train/mask'
pathTrainGT    = './img/train/normal'

pathTestColor = './img/test/color'
pathTestMask = './img/test/mask'

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _scan_png_files(folder):
    '''
    folder: 1.png 3.png 4.png 6.png 7.exr unknown.mpeg
    return: ['1.png', '3.png', '4.png']
    '''
    ext = '.png'
    ret = [_im_name for _im_name in os.listdir(folder) if _im_name.endswith(ext)]

    return ret

def genTRRecord_train(_imageDir, _maskDir, _gtDir):
    _image_names = _scan_png_files(_imageDir)
    _mask_names = _scan_png_files(_maskDir)
    _gt_names = _scan_png_files(_gtDir)

    _pred_diff_gt = set(_image_names).difference(_gt_names)
    assert len(_pred_diff_gt) == 0, \
        'No corresponding groundtruth file for the following files:\n' + '\n'.join(_pred_diff_gt)
    _pred_diff_mask = set(_image_names).difference(_mask_names)
    assert len(_pred_diff_mask) == 0, \
        'No corresponding mask file for the following files:\n' + '\n'.join(_pred_diff_mask)

    _cur_id = 0
    for _im_name in _image_names:
        print('Proccessing file {} - {}'.format(_im_name, _cur_id))
        writer = tf.python_io.TFRecordWriter('./trainTFRecords/' + _im_name.replace('.png','.tfrecords'))
        # image 0-255 float32
        _image = imageio.imread(os.path.join(_imageDir, _im_name)).astype(np.float32)
        # mask 0/255 float32
        _mask = imageio.imread(os.path.join(_maskDir, _im_name)).astype(np.float32) # Greyscale image
        # gt 0-255 float32
        _gt = imageio.imread(os.path.join(_gtDir, _im_name)).astype(np.float32)

        feature = {
            'image': _bytes_feature(tf.compat.as_bytes(_image.tostring())),
            'mask': _bytes_feature(tf.compat.as_bytes(_mask.tostring())),
            'gt': _bytes_feature(tf.compat.as_bytes(_gt.tostring()))
        }

        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
        _cur_id = _cur_id+1
        writer.close()

    sys.stdout.flush()

def genTRRecord_test(_imageDir, _maskDir):
    _image_names = _scan_png_files(_imageDir)
    _mask_names = _scan_png_files(_maskDir)

    _pred_diff_mask = set(_image_names).difference(_mask_names)
    assert len(_pred_diff_mask) == 0, \
        'No corresponding mask file for the following files:\n' + '\n'.join(_pred_diff_mask)

    _cur_id = 0
    for _im_name in _image_names:
        print('Proccessing file {} - {}'.format(_im_name, _cur_id))
        writer = tf.python_io.TFRecordWriter('./testTFRecords/' + _im_name.replace('.png','.tfrecords'))
        # image 0-1 float32
        _image = imageio.imread(os.path.join(_imageDir, _im_name)).astype(np.float32)
        # mask 0/255 float32
        _mask = imageio.imread(os.path.join(_maskDir, _im_name)).astype(np.float32) # Greyscale image

        feature = {
            'image': _bytes_feature(tf.compat.as_bytes(_image.tostring())),
            'mask': _bytes_feature(tf.compat.as_bytes(_mask.tostring()))
        }

        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
        _cur_id = _cur_id+1
        writer.close()

    sys.stdout.flush()

genTRRecord_train(pathTrainColor, pathTrainMask, pathTrainGT)
genTRRecord_test(pathTestColor, pathTestMask)