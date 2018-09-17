#!/usr/bin/env python
from __future__ import division
import os
import tensorflow as tf
import numpy as np
import time
from tfrecord import read_and_decode, tfrecord2metafilename
import matplotlib.pyplot as plt

def tf_deg2rad(deg):
    pi_on_180 = 0.017453292519943295
    return deg * pi_on_180

def affine_shift_transformation(img, tx=0., ty=0.):
  
  shift_trans = [1., 0., tx, 0., 1., ty, 0., 0.]

  img = tf.contrib.image.transform(img, shift_trans, interpolation='BILINEAR')

  return img

def affine_zoom_transformation(img, zx=0., zy=0.):
  
  zoom_trans = [zx, 0., 0., 0., zy, 0., 0., 0.]

  img = tf.contrib.image.transform(img, zoom_trans, interpolation='BILINEAR')

  return img

def affine_shear_transformation(img, shear=0.):
  
  shear = tf_deg2rad(shear)
  shear_trans = [1., -tf.sin(shear), 0., 0., tf.cos(shear), 0., 0., 0.]
  img = tf.contrib.image.transform(img, shear_trans, interpolation='BILINEAR')

  return img

def test_tfrecord():

  dic = { 
          'rand_crop':True,
          'rand_flip_left_right':False,
          'rand_flip_top_bottom':False,
          'zoom':False,
          'shear':False,
          'shift':False,
          'brightness_contrast':False,
          'rand_rotate':False,
          'elastic_deformation':False,

         }

  label_dic = {0: 'normal', 1: 'abnormal'}

  main_dir = '/Users/andreas/Desktop/tfrecords'
  meta = np.load(os.path.join(main_dir, 'val_meta.npz'))
  fn = os.path.join(main_dir, 'val.tfrecords')

  print 'Using train tfrecords: {0} | {1} images'.format(fn, len(meta['labels']))
  filename_queue = tf.train.string_input_producer([fn], num_epochs=1)

  model_dims = [360, 360, 1]
  batch_size = 2

  img, t_l, f_p = read_and_decode(filename_queue = filename_queue,
                                  img_dims = [400, 400, 1],
                                  model_dims = model_dims,
                                  size_of_batch = batch_size,
                                  augmentations_dic = dic,
                                  num_of_threads = 1,
                                  shuffle = False)

  shift_img = affine_shear_transformation(img, -15.)
  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

    sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
      while not coord.should_stop():

        np_img, np_shift_img, np_t_l, np_f_p = sess.run([img, shift_img, t_l, f_p])

        # np_img, np_t_l, np_f_p = sess.run([img, t_l, f_p])
        # print "{0}: label: {1}".format(np_f_p, label_dic[np_t_l[0]])
        # plt.imshow(np.squeeze(np_img))
        # plt.pause(0.2)

        plt.subplot(221)
        plt.imshow(np.squeeze(np_img[0,:,:,:]))
        plt.axis('off')

        plt.subplot(222)
        plt.imshow(np.squeeze(np_shift_img[0,:,:,:]))
        plt.axis('off')

        plt.subplot(223)
        plt.imshow(np.squeeze(np_img[1,:,:,:]))
        plt.axis('off')

        plt.subplot(224)
        plt.imshow(np.squeeze(np_shift_img[1,:,:,:]))
        plt.axis('off')
        plt.pause(3)

    except tf.errors.OutOfRangeError:
      print 'Done'
    finally:
      coord.request_stop()
    coord.join(threads)
    sess.close()

if __name__ == '__main__':
  test_tfrecord()
  