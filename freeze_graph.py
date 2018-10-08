import tensorflow as tf
import os
import tensorflow.contrib.image
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from tfrecord import  imagenet_preprocessing, tfrecord2metafilename, read_and_decode
import models.resnet_v2 as resnet_v2
import models.densenet as densenet
from tensorflow.python.framework import graph_util
from tensorflow.contrib import slim
from config import XRAYconfig

def save_checkpoint():
  config = XRAYconfig() 

  images = tf.placeholder(tf.float32, shape=[None,224,224,3], name='inputs')
  labels = tf.placeholder(tf.int32, shape=[None], name='labels')

  # with tf.variable_scope('resnet_v2_50') as resnet_scope:
  #   processed_image = imagenet_preprocessing(images)
  #   with slim.arg_scope(resnet_v2.resnet_arg_scope()):
  #     target_logits, _ = resnet_v2.resnet_v2_50(inputs = processed_image,
  #                                                  num_classes = config.output_shape, 
  #                                                  scope = resnet_scope,
  #                                                  is_training=False)

  with tf.variable_scope('densenet121') as densenet_scope:
    processed_image = imagenet_preprocessing(images)
    with slim.arg_scope(densenet.densenet_arg_scope(weight_decay = config.l2_reg, batch_norm_decay = config.batch_norm_decay, batch_norm_epsilon = config.batch_norm_epsilon)):
      target_logits, _ = densenet.densenet121(inputs = processed_image,
                                                   num_classes = config.output_shape, 
                                                   is_training=False,
                                                   scope = densenet_scope)

    oh_enc = tf.one_hot(labels, config.output_shape)
    # logits = target_logits * oh_enc
    masked_logits = tf.multiply(target_logits, oh_enc, name='masked_logits')
    # gradient = tf.gradients(logits, processed_image, name='gradients')
    prob = tf.nn.softmax(target_logits, name='probability')

  saver = tf.train.Saver() 
  restorer = tf.train.Saver()

  sess = tf.Session()
  sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
  restorer.restore(sess, config.restore_checkpoint)
  save_path = saver.save(sess, "./model_to_freeze.ckpt")
  print("Model saved in path: %s" % save_path)
  sess.close()

def freeze_graph():

  checkpoint =  "model_to_freeze.ckpt"

  saver = tf.train.import_meta_graph('model_to_freeze.ckpt.meta', clear_devices=True)
  graph = tf.get_default_graph()
  input_graph_def = graph.as_graph_def()
  print "Variables stored in checkpoint:"
  print_tensors_in_checkpoint_file(file_name=checkpoint, tensor_name='', all_tensors='')
  sess = tf.Session()
  saver.restore(sess, checkpoint)

  output_node_names = "densenet121/masked_logits,densenet121/probability"
  output_graph_def = graph_util.convert_variables_to_constants(
              sess, # The session
              input_graph_def, # input_graph_def is useful for retrieving the nodes 
              output_node_names.split(",")  
  )

  output_graph="frozen_graph.pb"
  with tf.gfile.GFile(output_graph, "wb") as f:
    f.write(output_graph_def.SerializeToString())
   
  sess.close()
  os.remove('model_to_freeze.ckpt.meta')
  os.remove('model_to_freeze.ckpt.index')
  os.remove('model_to_freeze.ckpt.data-00000-of-00001')

if __name__ == '__main__':
  save_checkpoint()
  freeze_graph()
