#!/usr/bin/env python
from __future__ import division
import argparse
import os
import tensorflow as tf
import datetime
import numpy as np
import time
from config import XRAYconfig
from tensorflow.contrib import slim
import models.se_resnet as se_resnet
from operator import add
from sklearn import metrics
import models.resnet_v2 as resnet_v2
import models.densenet as densenet
from data_utils import tfrecord2metafilename, read_and_decode, imagenet_preprocessing
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import math

def test_resnet(device, dataset, checkpoint):
  """
  Computes accuracy for the test dataset
  Input: gpu device 
  Output: None
  """
  os.environ['CUDA_VISIBLE_DEVICES'] = str(device) # use nvidia-smi to see available options '0' means first gpu
  config = XRAYconfig() # loads pathology configuration defined in vgg_config

  if dataset == 'val':  
    print "Using val..." 
    config.test_fn = os.path.join(config.main_dir, 'tfrecords/val.tfrecords')
  elif dataset == 'test':
    print "Using test..." 
    config.test_fn = os.path.join(config.main_dir, 'tfrecords/test.tfrecords')
  else:
    config.test_fn = None

  config.test_checkpoint = checkpoint
  print "Loading checkpoint: {0}".format(checkpoint)

  batch_size = 64
  # loading test data
  test_meta = np.load(tfrecord2metafilename(config.test_fn))
  print 'Using {0} tfrecords: {1} | {2} images'.format(dataset, config.test_fn, len(test_meta['labels']))
  test_filename_queue = tf.train.string_input_producer([config.test_fn] , num_epochs=1) # 1 epoch, passing through the
                                                                                          # the dataset once

  test_img, test_t_l, _, test_f_p  = read_and_decode(filename_queue = test_filename_queue,
                                           img_dims = config.input_image_size,
                                           model_dims = config.model_image_size,
                                           size_of_batch = batch_size,
                                           augmentations_dic = config.val_augmentations_dic,
                                           num_of_threads = 2,
                                           shuffle = False)

  # with tf.variable_scope('resnet_v2_50') as resnet_scope:
  #   test_img = imagenet_preprocessing(test_img)
  #   with slim.arg_scope(resnet_v2.resnet_arg_scope(weight_decay = config.l2_reg)):
  #     test_target_logits, _ = resnet_v2.resnet_v2_50(inputs = test_img,
  #                                                      num_classes = config.output_shape, 
  #                                                      scope = resnet_scope,
  #                                                      is_training = False)

  # with tf.variable_scope('resnet_v2_101') as resnet_scope:
  #   test_img = imagenet_preprocessing(test_img)
  #   with slim.arg_scope(se_resnet.resnet_arg_scope(weight_decay = config.l2_reg, batch_norm_decay = config.batch_norm_decay, batch_norm_epsilon = config.batch_norm_epsilon)):
  #     test_target_logits, _ = se_resnet.se_resnet_101(inputs = test_img,
  #                                                  num_classes = config.output_shape, 
  #                                                  scope = resnet_scope,
  #                                                  is_training=False)

  with tf.variable_scope('densenet121') as densenet_scope:
    test_img = imagenet_preprocessing(test_img)
    with slim.arg_scope(densenet.densenet_arg_scope(weight_decay = config.l2_reg, batch_norm_decay = config.batch_norm_decay, batch_norm_epsilon = config.batch_norm_epsilon)):
      test_target_logits, _ = densenet.densenet121(inputs = test_img,
                                                   num_classes = config.output_shape, 
                                                   is_training=False,
                                                   scope = densenet_scope) 
          
  target_prob = tf.nn.softmax(test_target_logits)
  prob_and_label_files = [target_prob,  test_t_l, test_f_p]
  restorer = tf.train.Saver()
  print "Variables stored in checkpoint:"
  print_tensors_in_checkpoint_file(file_name=config.test_checkpoint, tensor_name='', all_tensors='')
  
  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    
    sess.run(tf.group(tf.global_variables_initializer(),tf.local_variables_initializer()))

    restorer.restore(sess, config.test_checkpoint)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    all_predictions_target = []
    all_labels = []
    all_files = []

    study_dic = {}

    batch_num = 1
    try:
      print "Total number of batch iterations needed: {0}".format(int(math.ceil(len(test_meta['labels'])/batch_size)))
      while not coord.should_stop():
         
        np_prob_and_label_files = sess.run(prob_and_label_files)

        target_probs = np_prob_and_label_files[0]
        labels = np_prob_and_label_files[1] 
        files = np_prob_and_label_files[2]
        
        all_labels += list(labels) 
        all_files += list(files)
        preds =  list(np.argmax(target_probs, axis=1)) 
        all_predictions_target +=preds
        
        for f, p in zip(files, target_probs):

          split_f = f.split('/')
          study_key = split_f[-4] + '_' + split_f[-3] +'_' + split_f[-2]

          if study_key not in study_dic:
            study_dic[study_key] = [p]
          else:
            study_dic[study_key].append(p)

        print "evaluating current batch number: {0}".format(batch_num)
        batch_num +=1
     
    except tf.errors.OutOfRangeError:
      print "Per Image results:"
      print "{0} accuracy: {1:.2f}".format(dataset, (metrics.accuracy_score(all_labels, all_predictions_target)*100))
      print "{0} cohen kappa: {1:.2f}".format(dataset,(metrics.cohen_kappa_score(all_labels, all_predictions_target)*100))
      print
    finally:
      coord.request_stop()  
    coord.join(threads) 

  final_preds = []
  gold_standard = []

  dic_mapping = {'negative': 0, 'positive': 1}

  for k in study_dic.keys():
    prediction_array = study_dic[k] 
    final_pred = [0, 0]

    for p in prediction_array:
      final_pred[0] += p[0]
      final_pred[1] += p[1]

    final_pred = np.array(final_pred) / len(prediction_array)
    final_preds.append(np.argmax(final_pred))

    label = k.split('_')[-1]
    gold_standard.append(dic_mapping[label])

  print "Per Study results:"
  print "{0} accuracy: {1:.2f}".format(dataset, (metrics.accuracy_score(gold_standard, final_preds)*100))
  print "{0} cohen kappa: {1:.2f}".format(dataset,(metrics.cohen_kappa_score(gold_standard, final_preds)*100))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("device")
  parser.add_argument("dataset")
  parser.add_argument("checkpoint")
  args = parser.parse_args()

  test_resnet(args.device, args.dataset, args.checkpoint)
    