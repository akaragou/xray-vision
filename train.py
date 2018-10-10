#!/usr/bin/env python
from __future__ import division
import os
import tensorflow as tf
import argparse
import datetime
import numpy as np
import time
# import models.resnet_v2 as resnet_v2
import models.se_resnet as se_resnet
import models.densenet as densenet
import models.unet_preprocess as unet_preprocess
from tensorflow.contrib import slim
from config import XRAYconfig
from data_utils import tfrecord2metafilename, read_and_decode, imagenet_preprocessing

def weighted_softmax_cross_entropy_with_logits(train_labels, train_logits, output_shape, weights_file):

  train_one_hot = tf.one_hot(train_labels, output_shape)
  class_weights = np.load(weights_file).astype(np.float32)
  tf_class_weights = tf.constant(class_weights)
  weight_map = tf.multiply(train_one_hot, tf_class_weights)
  weight_map = tf.reduce_sum(weight_map, axis=1)
  batch_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels = train_one_hot, logits = train_logits)
  weighted_batch_loss = tf.multiply(batch_loss, weight_map)

  loss = tf.reduce_mean(weighted_batch_loss)
  return loss

def print_model_variables():
    print "Model Variables:"
    for var in slim.get_model_variables():
        print var 

def train_resnet(device, model):
  """
  Loads training and validations tf records and trains resnet model and validates every number of fixed steps.
  Input: gpu device number 
  Output None
  """
  os.environ['CUDA_VISIBLE_DEVICES'] = str(device) # use nvidia-smi to see available options '0' means first gpu
  config = XRAYconfig() # loads pathology configuration defined in resnet_config
  # load training data

  train_meta = np.load(tfrecord2metafilename(config.train_fn))
  print('Using train tfrecords: {0} | {1} images'.format(config.train_fn, len(train_meta['labels'])))
  train_filename_queue = tf.train.string_input_producer(
  [config.train_fn], num_epochs=config.num_train_epochs)
  # load validation data
  val_meta = np.load(tfrecord2metafilename(config.val_fn))
  print('Using test tfrecords: {0} | {1} images'.format(config.val_fn, len(val_meta['labels'])))
  val_filename_queue = tf.train.string_input_producer(
  [config.val_fn], num_epochs=config.num_train_epochs)

  model_train_name = model 
  dt_stamp = time.strftime(model_train_name  + "_%Y_%m_%d_%H_%M_%S")
  out_dir = config.get_results_path(model_train_name, dt_stamp)
  summary_dir = config.get_summaries_path(model_train_name, dt_stamp)
  print('-'*60)
  print('Training model: {0}'.format(dt_stamp))
  print('-'*60)

  train_img, train_t_l, train_b_t, _ = read_and_decode(filename_queue = train_filename_queue,
                                           img_dims = config.input_image_size,
                                           model_dims = config.model_image_size,
                                           size_of_batch = config.train_batch_size,
                                           augmentations_dic = config.train_augmentations_dic,
                                           num_of_threads = 4,
                                           shuffle = True)

  val_img, val_t_l, val_b_t, _  = read_and_decode(filename_queue = val_filename_queue,
                                       img_dims = config.input_image_size,
                                       model_dims = config.model_image_size,
                                       size_of_batch = config.val_batch_size,
                                       augmentations_dic = config.val_augmentations_dic,
                                       num_of_threads = 4,
                                       shuffle = False)

  # summaries to use with tensorboard check https://www.tensorflow.org/get_started/summaries_and_tensorboard
  tf.summary.image('train images', train_img, max_outputs=10)
  tf.summary.image('validation images', val_img, max_outputs=10)

  # creating step op that counts the number of training steps
  step = tf.train.get_or_create_global_step()
  step_op = tf.assign(step, step+1)

  if model == 'se_resnet_101':
    print("Loading Resnet 101...")
    with tf.variable_scope('resnet_v2_101') as resnet_scope:
      with tf.name_scope('train') as train_scope:
        train_img = imagenet_preprocessing(train_img)
        with slim.arg_scope(se_resnet.resnet_arg_scope(weight_decay = config.l2_reg, batch_norm_decay = config.batch_norm_decay, batch_norm_epsilon = config.batch_norm_epsilon)):
          train_target_logits, _ = se_resnet.se_resnet_101(inputs = train_img,                                                               
                                                          num_classes = config.output_shape,
                                                          scope = resnet_scope,                              
                                                          is_training = True)

      resnet_scope.reuse_variables()  
      with tf.name_scope('val') as val_scope:
        val_img = imagenet_preprocessing(val_img)
        with slim.arg_scope(se_resnet.resnet_arg_scope(weight_decay = config.l2_reg, batch_norm_decay = config.batch_norm_decay, batch_norm_epsilon = config.batch_norm_epsilon)):
          val_target_logits, _ = se_resnet.se_resnet_101(inputs = val_img,
                                                       num_classes = config.output_shape, 
                                                       scope = resnet_scope,
                                                       is_training=False)
  elif model == 'densenet_121':
    print("Loading Densenet 121...")
    with tf.variable_scope('densenet121') as densenet_scope:
      with tf.name_scope('train') as train_scope:
        train_img = imagenet_preprocessing(train_img)
        with slim.arg_scope(densenet.densenet_arg_scope(weight_decay = config.l2_reg, batch_norm_decay = config.batch_norm_decay, batch_norm_epsilon = config.batch_norm_epsilon)):
          train_target_logits, _ = densenet.densenet121(inputs = train_img,                                                               
                                                          num_classes = config.output_shape,                      
                                                          is_training = True,
                                                          scope = densenet_scope)
      print_model_variables()
      densenet_scope.reuse_variables()  
      with tf.name_scope('val') as val_scope:
        val_img = imagenet_preprocessing(val_img)
        with slim.arg_scope(densenet.densenet_arg_scope(weight_decay = config.l2_reg, batch_norm_decay = config.batch_norm_decay, batch_norm_epsilon = config.batch_norm_epsilon)):
          val_target_logits, _ = densenet.densenet121(inputs = val_img,
                                                       num_classes = config.output_shape, 
                                                       is_training=False,
                                                       scope = densenet_scope)
  else: 
    raise Exception('Model not implemented! Options are resnet_50 and densenet_121')

  loss = weighted_softmax_cross_entropy_with_logits(train_t_l, train_target_logits, config.output_shape, 'target_class_weights.npy')

  tf.summary.scalar("loss", loss)

  lr = tf.train.exponential_decay(
          learning_rate = config.initial_learning_rate,
          global_step = step_op,
          decay_steps = config.decay_steps,
          decay_rate = config.learning_rate_decay_factor,
          staircase = True) # if staircase is True decay the learning rate at discrete intervals

  if config.optimizer == "adam":
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # used to update batch norm params. see https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization
    with tf.control_dependencies(update_ops):
      train_op = tf.train.AdamOptimizer(lr).minimize(loss)
  elif config.optimizer == "sgd":
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op =  tf.train.GradientDescentOptimizer(lr).minimize(loss)
  elif config.optimizer == "nestrov":
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op =  tf.train.MomentumOptimizer(lr, config.momentum, use_nesterov=True).minimize(loss)
  else:
      raise Exception("Not known optimizer! options are adam, sgd or nestrov")

  train_prob = tf.nn.softmax(train_target_logits)
  train_pred = tf.argmax(train_prob, 1)
 
  val_prob = tf.nn.softmax(val_target_logits)
  val_pred = tf.argmax(val_prob, 1)

  train_accuracy = tf.contrib.metrics.accuracy(train_pred, train_t_l)
  val_accuracy = tf.contrib.metrics.accuracy(val_pred , val_t_l)

  train_auc, train_auc_op = tf.metrics.auc(train_t_l, train_pred)
  val_auc, val_auc_op = tf.metrics.auc(val_t_l, val_pred)

  tf.summary.scalar("training accuracy", train_accuracy)
  tf.summary.scalar("validation accuracy", val_accuracy)
  tf.summary.scalar("training auc", train_auc)
  tf.summary.scalar("validation auc", val_auc)

  if config.restore:
    # adjusting variables to keep in the model
    # variables that are exluded will allow for transfer learning (normally fully connected layers are excluded)
    exclusions = [scope.strip() for scope in config.checkpoint_exclude_scopes]
    variables_to_restore = []
    for var in slim.get_model_variables():
      excluded = False
      for exclusion in exclusions:
        if var.op.name.startswith(exclusion):
          excluded = True
          break
      if not excluded:
        variables_to_restore.append(var)
    print("Restroing variables:")
    for var in variables_to_restore:
      print(var)
    restorer = tf.train.Saver(variables_to_restore)
  saver = tf.train.Saver(slim.get_model_variables(), max_to_keep=100)
  summary_op = tf.summary.merge_all()

  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

    sess.run(tf.group(tf.global_variables_initializer(),
         tf.local_variables_initializer()))

    if config.restore:
      restorer.restore(sess, config.model_path)

    summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    np.save(os.path.join(out_dir, 'training_config_file'), config)

    val_acc_max = 0

    try:

      while not coord.should_stop():
        
        start_time = time.time()

        step_count, loss_value, train_acc_value, lr_value, _ = sess.run([step_op, loss, train_accuracy,lr, train_op])
        sess.run(train_auc_op)
        train_auc_value = sess.run(train_auc)

        duration = time.time() - start_time
        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
        step_count = step_count - 1 

        if step_count % config.validate_every_num_steps == 0:
          it_val_acc = np.asarray([])
          for num_vals in range(config.num_batches_to_validate_over):
              # Validation accuracy as the average of n batches
            it_val_acc = np.append(it_val_acc, sess.run(val_accuracy))
            sess.run(val_auc_op)
            
          val_acc_value = it_val_acc.mean()
          val_auc_value = sess.run(val_auc)
          # Summaries
          summary_str = sess.run(summary_op)
          summary_writer.add_summary(summary_str, step_count)

          # Training status and validation accuracy
          msg = '{0}: step {1}, loss = {2:.4f} ({3:.2f} examples/sec; '\
              + '{4:.2f} sec/batch) | Training accuracy = {5:.4f} | Training AUC = {6:.4f} '\
              + '| Validation accuracy = {7:.4f} | Validation AUC = {8:.4f}| logdir = {9}'
          print(msg.format(
                datetime.datetime.now(), step_count, loss_value,
                (config.train_batch_size / duration), float(duration),
                train_acc_value, train_auc_value, val_acc_value, val_auc_value, summary_dir))
          print("Learning rate: {}".format(lr_value))
          # Save the model checkpoint if it's the best yet
          if val_acc_value >= val_acc_max:
            file_name = '{0}_{1}'.format(dt_stamp, step_count)
            saver.save( sess, config.get_checkpoint_filename(model_train_name, file_name))
            # Store the new max validation accuracy
            val_acc_max = val_acc_value

        else:
          # Training status
          msg = '{0}: step {1}, loss = {2:.4f} ({3:.2f} examples/sec; '\
              + '{4:.2f} sec/batch) | Training accuracy = {5:.4f} | Training AUC = {6:.4f}'
          print(msg.format(datetime.datetime.now(), step_count, loss_value,
                (config.train_batch_size / duration),
                float(duration), train_acc_value, train_auc_value))
        # End iteration

    except tf.errors.OutOfRangeError:
      print('Done training for {0} epochs, {1} steps.'.format(config.num_train_epochs, step_count))
    finally:
      coord.request_stop()
    coord.join(threads)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("device")
  parser.add_argument("model")
  args = parser.parse_args()
  train_resnet(args.device, args.model) 
