#!/usr/bin/env python
from __future__ import division
import os
import tensorflow as tf
import numpy as np
import time
from tfrecord import read_and_decode, tfrecord2metafilename
import matplotlib.pyplot as plt
import math
PI = tf.constant(math.pi)

def test_tfrecord():

  dic = { 
          'rand_crop':False,
          'rand_flip_left_right':False,
          'rand_flip_top_bottom':False,
          'zoom':True,
          'shear':True,
          'shift':False,
          'brightness_contrast':False,
          'rand_rotate':False,
          'elastic_deformation':False,
         }

  label_dic = {0: 'normal', 1: 'abnormal'}
  bone_dic = {0:'XR_ELBOW', 1:'XR_FINGER', 2:'XR_FOREARM', 3:'XR_HAND', 4:'XR_HUMERUS', 5:'XR_SHOULDER', 6:'XR_WRIST'}

  main_dir = '../tfrecords'
  meta = np.load(os.path.join(main_dir, 'XR_ELBOW_abnormal_meta.npz'))
  fn = os.path.join(main_dir, 'XR_ELBOW_abnormal.tfrecords')

  print('Using tfrecords: {0} | {1} images'.format(fn, len(meta['labels'])))
  filename_queue = tf.train.string_input_producer([fn], num_epochs=1)


  model_dims = [224, 224, 3]
  batch_size = 1

  XR_ELBOW_normal_img, XR_ELBOW_normal_t_l, XR_ELBOW_normal_b_t, XR_ELBOW_normal_f_p  = read_and_decode(filename_queue = filename_queue,
                                                                                                                img_dims = [256, 256, 3],
                                                                                                                model_dims = model_dims,
                                                                                                                size_of_batch = batch_size,
                                                                                                                augmentations_dic = dic,
                                                                                                                num_of_threads = 1,
                                                                                                                shuffle = False)

  meta = np.load(os.path.join(main_dir, 'XR_ELBOW_abnormal_meta.npz'))
  fn = os.path.join(main_dir, 'XR_ELBOW_abnormal.tfrecords')

  print('Using tfrecords: {0} | {1} images'.format(fn, len(meta['labels'])))
  filename_queue = tf.train.string_input_producer([fn], num_epochs=1)

  XR_ELBOW_abnormal_img, XR_ELBOW_abnormal_t_l, XR_ELBOW_abnormal_b_t, XR_ELBOW_abnormal_f_p  = read_and_decode(filename_queue = filename_queue,
                                                                                                                img_dims = [256, 256, 3],
                                                                                                                model_dims = model_dims,
                                                                                                                size_of_batch = batch_size,
                                                                                                                augmentations_dic = dic,
                                                                                                                num_of_threads = 1,
                                                                                                                shuffle = False)

  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

    sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
      while not coord.should_stop():

        # np_img, np_shift_img, np_t_l, np_f_p = sess.run([img, shift_img, t_l, f_p])

        # np_img, np_t_l, np_b_t, np_f_p = sess.run([img, t_l, b_t, f_p])
        # print("{0}: label: {1} || bone type: {2}").format(np_f_p, label_dic[np_t_l[0]], bone_dic[np_b_t[0]])
        # plt.imshow(np.squeeze(np_img))
        # plt.pause(0.2)

        normal_f_p, abnormal_f_p = sess.run([XR_ELBOW_normal_f_p, XR_ELBOW_abnormal_f_p])
        print normal_f_p
        print abnormal_f_p


    except tf.errors.OutOfRangeError:
      print('Done')
    finally:
      coord.request_stop()
    coord.join(threads)
    sess.close()

def test_metric():

  labels = np.array([[1,1,1,0],
                   [1,1,1,0],
                   [1,1,1,0],
                   [1,1,1,0]], dtype=np.uint8)

  predictions = np.array([[1,0,0,0],
                        [1,1,1,0],
                        [1,1,1,0],
                        [1,1,1,0]], dtype=np.uint8)

  n_batches = len(labels)
  graph = tf.Graph()
  with graph.as_default():
    # Placeholders to take in batches onf data
    tf_label = tf.placeholder(dtype=tf.int32, shape=[None])
    tf_prediction = tf.placeholder(dtype=tf.int32, shape=[None])

    # Define the metric and update operations
    # tf_metric, tf_metric_update = tf.contrib.metrics.cohen_kappa(tf_label,
    #                                                   tf_prediction, 2,
    #                                                   name="my_metric")
    tf_metric, tf_metric_update = tf.contrib.metrics.streaming_auc(tf_label,
                                                      tf_prediction,
                                                      name="my_metric")

    # Isolate the variables stored behind the scenes by the metric operation
    running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="my_metric")

    # Define initializer to initialize/reset running variables
    running_vars_initializer = tf.variables_initializer(var_list=running_vars)


  with tf.Session(graph=graph) as session:
    session.run(tf.global_variables_initializer())

    

    for i in range(n_batches):

      # reset the running variables
      session.run(running_vars_initializer)

      # Update the running variables on new batch of samples
      feed_dict={tf_label: labels[i], tf_prediction: predictions[i]}
      session.run(tf_metric_update, feed_dict=feed_dict)

      # Calculate the score
      score = session.run(tf_metric)
      print("TF METRIC: {0}".format(score))


if __name__ == '__main__':
  test_tfrecord()
  # test_metric()
  