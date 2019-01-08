#!/usr/bin/env python
from __future__ import division
import tensorflow as tf
import glob
import numpy as np
import os
import csv
from data_utils import create_tf_record
import time
import gc

def compute_and_store_weights(dic, label_type):
  """
  Computes and saves weights for abnormal 
  or normal classes or distribution of bone types
  Inputs: dic - dictionary of class to count
          label_type - type of label to compute class weigts for
  Outputs None
  """
  values = [dic[k] for k in range(len(dic.keys()))]
  f_i = [sum(values)/v for v in values]
  class_weights = [f/sum(f_i) for f in f_i]
  print label_type + ":"
  print "class ratios:", [v/sum(values) for v in values]
  print "class weights:", class_weights
  print
  np.save(label_type + '_class_weights.npy', class_weights)

def build_train_val_tfrecords(csv_data_dirs, main_tfrecords_dir, bone_dic):
  """
  Build tfrecords for training and validation images and labels
  Inputs: csv_data_dirs - directory where train and val data is located
          main_tfrecords_dir - directory to store tfrecords in
          bone_dic - dictionary for bone types
  Outputs: None
  """
  patient_csv_file = os.path.join(csv_data_dirs, 'train_labeled_studies.csv')
  val_csv_file = os.path.join(csv_data_dirs, 'val.csv')

  val_patient_files = []
  with open(val_csv_file , 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
      val_patient_files.append(row[0])

  train_files = []
  train_target_labels = []
  train_bone_type_labels = []

  val_files = []
  val_target_labels = []
  val_bone_type_labels = []

  with open(patient_csv_file , 'rb') as f:
    reader = csv.reader(f)
    for row in reader:

      bone_type = row[0].split('/')[2]

      if row[0] not in val_patient_files:
        files = glob.glob(os.path.join('../mura/',row[0]) + '/*.png')
        train_files = train_files + files 
        train_target_labels = train_target_labels + [int(row[1]) for _ in range(len(files))]
        train_bone_type_labels = train_bone_type_labels + [bone_dic[bone_type] for _ in range(len(files))]
      else:
        files = glob.glob(os.path.join('../mura/',row[0]) + '/*.png')
        val_files = val_files + files 
        val_target_labels = val_target_labels + [int(row[1]) for _ in range(len(files))]
        val_bone_type_labels = val_bone_type_labels + [bone_dic[bone_type] for _ in range(len(files))]

  calculate_target_and_bone_ratios(train_bone_type_labels, train_target_labels)
  create_tf_record(os.path.join(main_tfrecords_dir, 'train.tfrecords'), train_files, train_target_labels, train_bone_type_labels)
  create_tf_record(os.path.join(main_tfrecords_dir, 'val.tfrecords'), val_files, val_target_labels, val_bone_type_labels)

def build_test_tfrecords(csv_data_dirs, main_tfrecords_dir, bone_dic):
  """
  Build tfrecords for test images and labels
  Inputs: csv_data_dirs - directory where train and val data is located
          main_tfrecords_dir - directory to store tfrecords in
          bone_dic - dictionary for bone types
  Outputs: None
  """
  patient_csv_file = os.path.join(csv_data_dirs, 'valid_labeled_studies.csv')

  test_files = []
  test_labels = []
  test_bone_type_labels = []

  with open(patient_csv_file , 'rb') as f:
    reader = csv.reader(f)
    for row in reader:

      files = glob.glob(os.path.join('../mura/',row[0]) + '/*.png')
      test_files = test_files + files 
      test_labels = test_labels + [int(row[1]) for _ in range(len(files))]

      bone_type = row[0].split('/')[2]
      test_bone_type_labels = test_bone_type_labels + [bone_dic[bone_type] for _ in range(len(files))]

  create_tf_record(os.path.join(main_tfrecords_dir, 'test.tfrecords'), test_files, test_labels, test_bone_type_labels)

if __name__ == '__main__':

  csv_data_dirs = '../mura/MURA-v1.1/'
  main_tfrecords_dir = '../tfrecords/'

  bone_dic = {'XR_ELBOW':0, 'XR_FINGER':1, 'XR_FOREARM':2, 'XR_HAND':3, 'XR_HUMERUS':4, 'XR_SHOULDER':5, 'XR_WRIST':6}

  build_train_val_tfrecords(csv_data_dirs, main_tfrecords_dir, bone_dic)
  build_test_tfrecords(csv_data_dirs, main_tfrecords_dir, bone_dic)
