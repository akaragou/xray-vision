#!/usr/bin/env python
from __future__ import division
import os
import tensorflow as tf
import numpy as np
import random
import argparse
import glob
from scipy import misc
from tensorflow.contrib import slim
import models.se_resnet as se_resnet
import models.resnet_v2 as resnet_v2
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.misc import imsave
from PIL import Image, ImageOps
from config import XRAYconfig
import cv2
from tfrecord import  imagenet_preprocessing, tfrecord2metafilename, read_and_decode
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file                                                                               

def maintain_aspec_ratio_resize(img, desired_size=256):

      old_size = img.size  # old_size[0] is in (width, height) format

      ratio = float(desired_size)/max(old_size)
      new_size = tuple([int(x*ratio) for x in old_size])

      img = img.resize(new_size, Image.ANTIALIAS)

      new_img = Image.new("RGB", (desired_size, desired_size))
      new_img.paste(img, ((desired_size-new_size[0])//2,
                        (desired_size-new_size[1])//2))

      new_img = np.array(new_img)
      return new_img

def centeredCrop(img, new_height, new_width):

   width =  np.size(img,1)
   height =  np.size(img,0)
   left = int(np.ceil((width - new_width)/2.))
   top = int(np.ceil((height - new_height)/2.))
   right = int(np.floor((width + new_width)/2.))
   bottom = int(np.floor((height + new_height)/2.))

   print left
   print top
   cImg = img[top:bottom, left:right]
   return cImg

def image_batcher(start, num_batches, images, labels, batch_size):
  """Placeholder image/label batch loader.
  Inputs: start - a start index, usually set at 0
          num_batches - total number of batches_of_batches
          images - an array of filepointers to images
          labels - an array of labels corresponding to images
          batch_size - size of batch that will be fed 
  Output: yields a generator of images, labels and filepointers
  """
  for _ in range(num_batches):
      next_image_batch = images[start:start + batch_size]
      image_stack = []
      if labels is None:
          label_stack = None
      else:
          label_stack = labels[start:start + batch_size]
      for f in next_image_batch:

          img = misc.imread(f).astype(np.float32)/255.0
          image_stack += [img[None, :, :, :]]

      start += batch_size
      yield np.concatenate(image_stack, axis=0), label_stack, next_image_batch

def feature_extract_for_SmoothGrad(device):
  """
  Loads test images and computes SmoothGrad for each image with options for plotting and storing masked images 
  Input: gpu device number 
  Output: None
  """

  os.environ['CUDA_VISIBLE_DEVICES'] = str(device) # use nvidia-smi to see available options '0' means first gpu
  config = XRAYconfig() # loads pathology configuration defined in vgg_config

  images = tf.placeholder(tf.float32, shape=[None,224,224,3], name='inputs')
  labels = tf.placeholder(tf.int32, shape=[None], name='labels')

  with tf.variable_scope('resnet_v2_50') as resnet_scope:
    processed_image = imagenet_preprocessing(images)
    with slim.arg_scope(resnet_v2.resnet_arg_scope(weight_decay = config.l2_reg, batch_norm_decay = config.batch_norm_decay, batch_norm_epsilon = config.batch_norm_epsilon)):
      target_logits, _ = resnet_v2.resnet_v2_50(inputs = processed_image,
                                                   num_classes = config.output_shape, 
                                                   scope = resnet_scope,
                                                   is_training=False)

      oh_enc = tf.one_hot(labels, config.output_shape)
      logits = target_logits * oh_enc
      gradient = tf.gradients(logits, processed_image, name='gradients')
      prob = tf.nn.softmax(target_logits, name='probability')
  
  saver = tf.train.Saver() 
  restorer = tf.train.Saver()
  print "Variables stored in checpoint:"
  print_tensors_in_checkpoint_file(file_name=config.restore_checkpoint, tensor_name='',all_tensors='')

  img = Image.open('elbow.png')
  img = maintain_aspec_ratio_resize(img)
  img = centeredCrop(img, 224, 224)
  img = img[np.newaxis,:,:,:]
  img = img/255.0
  label_batch = [1]

  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

    sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
    restorer.restore(sess, config.restore_checkpoint) # restotring model checkpoints

    grad_im_list = []
    for _ in range(config.num_iter):
      eps = np.random.normal(config.mu, config.sd, config.model_image_size)
      
      image_eps = img + eps
      
      grad_im_list += [sess.run(gradient,feed_dict={images: image_eps,
                                                      labels: label_batch})]

    M = np.mean(np.concatenate(grad_im_list), axis=0) # M is the smooth grad image, checkout https://arxiv.org/abs/1706.03825 
    # normalizing by taking the absolute value per pixel 
    # and the suming across all three chanels 
    abs_M = np.abs(M)
    sum_M = np.sum(abs_M, axis=3)
    # prediction for current batch image
    predicted = sess.run(prob, feed_dict={images: img,
                                          labels: label_batch})

    mask = np.squeeze(sum_M)
    thres = np.percentile(mask.ravel(), 99)
    idx = mask[:,:] < thres
    mask[idx] = 0
    kernel = np.ones((5,5),np.float32)/25
    mask = cv2.filter2D(mask,-1,kernel)
    plt.figure()
    plt.imshow(np.squeeze(img), interpolation='none')
    plt.imshow(mask, interpolation='none', alpha=0.3)
    plt.axis('off')
    plt.savefig('out.png', transparent = True, bbox_inches='tight', pad_inches=0)
    print predicted

    save_path = saver.save(sess, "./model_to_freeze.ckpt")
    print("Model saved in path: %s" % save_path)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("device")
  args = parser.parse_args()
  feature_extract_for_SmoothGrad(args.device)
  # create_mask_filepaths()   