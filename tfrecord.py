#!/usr/bin/env python
from __future__ import division
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from scipy import misc
from tqdm import tqdm
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import repeat
import math
import cv2

IMAGENET_MEANS = [103.939, 116.779, 123.68]
PI = tf.constant(math.pi)

def _bytes_feature(value):
  return tf.train.Feature(bytes_list = tf.train.BytesList(value=[value]))

def _int64_feature(value):
  return tf.train.Feature(int64_list = tf.train.Int64List(value=[value]))

def _float_feature(value):
  return tf.train.Feature(float_list = tf.train.FloatList(value=[value]))

def tf_deg2rad(deg):
  pi_on_180 = 0.017453292519943295
  return deg * pi_on_180

def draw_grid(img, grid_size): 
  for i in range(0, np.shape(img)[1], grid_size):
    cv2.line(img, (i, 0), (i, np.shape(img)[0]), color=(0,0,0),thickness=2)
  for j in range(0, np.shape(img)[0], grid_size):
    cv2.line(img, (0, j), (np.shape(img)[1], j), color=(0,0,0),thickness=2)
  return img

def tfrecord2metafilename(tfrecord_filename):
  base, ext = os.path.splitext(tfrecord_filename)
  return base + '_meta.npz'

def imagenet_preprocessing(image_rgb):

  image_rgb_scaled = image_rgb * 255.0
  red, green, blue = tf.split(num_or_size_splits=3, axis=3, value=image_rgb_scaled)
  assert red.get_shape().as_list()[1:] == [224, 224, 1]
  assert green.get_shape().as_list()[1:] == [224, 224, 1]
  assert blue.get_shape().as_list()[1:] == [224, 224, 1]
  image_bgr = tf.concat(values = [
      blue - IMAGENET_MEANS[0],
      green - IMAGENET_MEANS[1],
      red - IMAGENET_MEANS[2],
      ], axis=3)
  assert image_bgr.get_shape().as_list()[1:] == [224, 224, 3], image_bgr.get_shape().as_list()
  return image_bgr

def elastic_deformation(img, model_dims, size_of_batch, mean = 0.0, sigma = 1.0, ksize = 180, alpha = 6.0):

  X = tf.random_uniform([model_dims[0], model_dims[1]])*2 - 1
  Y = tf.random_uniform([model_dims[0], model_dims[1]])*2 - 1
  X = tf.reshape(X, [1, model_dims[0], model_dims[1], 1])
  Y = tf.reshape(Y, [1, model_dims[0], model_dims[1], 1])

  x = tf.linspace(-3.0, 3.0, ksize)
  z = ((1.0 / (sigma * tf.sqrt(2.0 * PI))) * tf.exp(tf.negative(tf.pow(x - mean, 2.0) / (2.0 * tf.pow(sigma, 2.0)))))
  z_2d = tf.matmul(tf.reshape(z, [ksize, 1]), tf.reshape(z, [1, ksize]))
  z_4d = tf.reshape(z_2d, [ksize, ksize, 1, 1])

  X_convolved = tf.nn.conv2d(X, z_4d, strides=[1, 1, 1, 1], padding='SAME')
  Y_convolved = tf.nn.conv2d(Y, z_4d, strides=[1, 1, 1, 1], padding='SAME')

  X_convolved = (X_convolved / tf.reduce_max(X_convolved))*alpha
  Y_convolved = (Y_convolved / tf.reduce_max(Y_convolved))*alpha

  trans = tf.stack([X_convolved,Y_convolved], axis=-1)
  trans = tf.reshape(trans, [-1])

  batch_trans = tf.tile(trans, [size_of_batch])
  batch_trans = tf.reshape(batch_trans, [size_of_batch, model_dims[0], model_dims[1] ,2])

  img = tf.reshape(img, [size_of_batch, model_dims[0], model_dims[1], model_dims[2]])
  img = tf.contrib.image.dense_image_warp(img, batch_trans)

  return img

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

def distort_brightness_constrast(image, ordering=0):

  if ordering == 0:
    image = tf.image.random_brightness(image, max_delta=32. / 255.)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
  else:
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    image = tf.image.random_brightness(image, max_delta=32. / 255.)
  return tf.clip_by_value(image, 0.0, 1.0)

def encode(img_path, target_label, convert_to_grayscale):

  if convert_to_grayscale:
    img = misc.imread(img_path, mode='L').astype(np.float32)
    img = draw_grid(img, 50)

  else:
    img = np.array(Image.open(img_path))
    if len(np.shape(img)) == 2:
      three_channel_img = np.stack([img, img, img], axis=-1)
      img = three_channel_img

  img = misc.imresize(img, (400, 400))
  img_raw = img.astype(np.uint16).tostring()
  path_raw = img_path.encode('utf-8')

  example = tf.train.Example(features=tf.train.Features(feature={
          'image_raw': _bytes_feature(img_raw),
          'file_path': _bytes_feature(path_raw),
          'target_label':_int64_feature(int(target_label)),
          }))
  return example

def create_tf_record(tfrecords_filename, file_pointers, target_labels, convert_to_grayscale=False):
    
  writer = tf.python_io.TFRecordWriter(tfrecords_filename)
  print '%d files in %d categories' % (len(np.unique(file_pointers)), len(np.unique(target_labels)))
  with ProcessPoolExecutor(4) as executor:
    futures = [executor.submit(encode, f, t_l, convert_to_grayscale) for f, t_l in zip(file_pointers, target_labels)]
    kwargs = {
        'total': len(futures),
        'unit': 'it',
        'unit_scale': True,
        'leave': True
    }

    for f in tqdm(as_completed(futures), **kwargs):
        pass
    print "Done loading futures!"
    print "Writing examples..."
    for i in tqdm(range(len(futures))):
      try:
          example = futures[i].result()
          writer.write(example.SerializeToString())
      except Exception as e:
          print "Failed to write example!"
  meta = tfrecord2metafilename(tfrecords_filename)
  np.savez(meta, file_pointers=file_pointers, labels=target_labels, output_pointer=tfrecords_filename)
  print '-' * 100
  print 'Generated tfrecord at %s' % tfrecords_filename
  print '-' * 100

def read_and_decode(filename_queue=None, img_dims=[400,400,1], model_dims=[360,360,1], size_of_batch=32,\
                    augmentations_dic=None, num_of_threads=1, shuffle=True):

  reader = tf.TFRecordReader()

  _, serialized_example = reader.read(filename_queue)
  
  features = tf.parse_single_example(
    serialized_example,
  
    features={
      'image_raw': tf.FixedLenFeature([], tf.string),
      'file_path': tf.FixedLenFeature([], tf.string),
      'target_label': tf.FixedLenFeature([], tf.int64), 

      })

  image = tf.decode_raw(features['image_raw'], tf.uint16)
  target_label = tf.cast(features['target_label'], tf.int32)
  file_path = tf.cast(features['file_path'], tf.string)

  image = tf.reshape(image, img_dims)
  image = tf.cast(image, tf.float32)

  image = image / 255.0
  
  if augmentations_dic['rand_flip_left_right']:
    image = tf.image.random_flip_left_right(image)

  if augmentations_dic['rand_flip_top_bottom']:
    image = tf.image.random_flip_up_down(image)

  if augmentations_dic['zoom']:
    zx = tf.random_uniform(0.7, 1.1)
    zy = tf.random_uniform(0.7, 1.1)
    image = affine_zoom_transformation(image, zx, zy)

  if augmentations_dic['shear']:
    image = affine_shear_transformation(image, shear=0.)

  if augmentations_dic['shift']:
    image = affine_shift_transformation(image, zx=0., zy=0.)

  if augmentations_dic['brightness_contrast']:  
    image = distort_brightness_constrast(image,  order_num)

  if augmentations_dic['rand_rotate']:
    elems = tf.cast(tf.convert_to_tensor(np.deg2rad(np.array(range(-30,31)))), dtype=tf.float32)
    sample = tf.squeeze(tf.multinomial(tf.log([(1.0/np.repeat(71, 71)).tolist()]), 1)) 
    random_angle = elems[tf.cast(sample, tf.int32)]
    image = tf.contrib.image.rotate(image, random_angle)  

  if augmentations_dic['rand_crop']:
    image = tf.random_crop(image, model_dims)
  else:
    image = tf.image.resize_image_with_crop_or_pad(image, model_dims[0],\
                                                   model_dims[1])

  if shuffle:
    img, t_l, f_p = tf.train.shuffle_batch([image, target_label, file_path],
                                                   batch_size=size_of_batch,
                                                   capacity=1000 + 3 * size_of_batch,
                                                   min_after_dequeue=1000,
                                                   num_threads=num_of_threads)
  else:
    img, t_l, f_p = tf.train.batch([image, target_label, file_path],
                                         batch_size=size_of_batch,
                                         capacity=5000,
                                         allow_smaller_final_batch=True,
                                         num_threads=num_of_threads)

  if augmentations_dic['elastic_deformation']:
    img = elastic_deformation(img, model_dims, size_of_batch)
    
  return  img, t_l, f_p

