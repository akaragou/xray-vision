#!/usr/bin/env python
from __future__ import division
import tensorflow as tf
import glob
import numpy as np
import os
import csv
from tfrecord import create_tf_record
import time
import gc

def create_tf_record_by_bone_type(bone_type, bone_dic_array):

  target_labels_normal = [0 for _ in range(len(bone_dic_array[bone_type][0]))]
  bone_type_labels_normal = [bone_dic[bone_type] for _ in range(len(bone_dic_array[bone_type][0]))]

  target_labels_abnormal = [1 for _ in range(len(bone_dic_array[bone_type][1]))]
  bone_type_labels_abnormal = [bone_dic[bone_type] for _ in range(len(bone_dic_array[bone_type][1]))]

  create_tf_record(os.path.join(main_tfrecords_dir, bone_type + '_normal.tfrecords'), bone_dic_array[bone_type][0], target_labels_normal, bone_type_labels_normal)
  create_tf_record(os.path.join(main_tfrecords_dir, bone_type + '_abnormal.tfrecords'), bone_dic_array[bone_type][1], target_labels_abnormal, bone_type_labels_abnormal)

def compute_and_store_weights(dic, label_type):
  values = [dic[k] for k in range(len(dic.keys()))]
  f_i = [sum(values)/v for v in values]
  class_weights = [f/sum(f_i) for f in f_i]
  print label_type + ":"
  print "class ratios:", [v/sum(values) for v in values]
  print "class weights:", class_weights
  print
  np.save(label_type + '_class_weights.npy', class_weights)

def calculate_target_and_bone_ratios(bone_types, target_labels):

    bone_count_dic = {}
    label_count_dic = {}

    for b_t, l in zip(bone_types, target_labels):

      if b_t not in bone_count_dic:
        bone_count_dic[b_t] = 1
      else:
        bone_count_dic[b_t] += 1

      if l not in label_count_dic:
        label_count_dic[l] = 1
      else:
        label_count_dic[l] += 1

    compute_and_store_weights(label_count_dic, 'target')
    compute_and_store_weights(bone_count_dic, 'bone_type')

def compute_mean_grey(file):

  img = misc.imread(file, mode='L')
  img_mean = np.mean(img)

  return img_mean

def calculate_img_stats(files):

  grey_mean = []

  with ProcessPoolExecutor(4) as executor:
    futures = [executor.submit(compute_mean_grey, f) for f in files]
    kwargs = {
          'total': len(futures),
          'unit': 'it',
          'unit_scale': True,
          'leave': True
    }

    for f in tqdm(as_completed(futures), **kwargs):
        pass
    print("Done loading futures!")
    for i in tqdm(range(len(futures))):
      try:
        example = futures[i].result()
        grey_mean.append(example)
      except Exception as e:
        print("Failed to compute means")

  print("grey mean: {0} || grey std: {1}".format(np.mean(grey_mean), np.std(grey_mean)))

def build_train_tfrecords_by_bone_type(csv_data_dirs, main_tfrecords_dir, bone_dic):
    
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

  bone_dic_array = {'XR_ELBOW':[[],[]], 'XR_WRIST':[[], []]}

  with open(patient_csv_file , 'rb') as f:
    reader = csv.reader(f)
    for row in reader:

      bone_type = row[0].split('/')[2]

      if row[0] not in val_patient_files:
        files = glob.glob(os.path.join('../mura/',row[0]) + '/*.png')
        train_files = train_files + files 
        train_target_labels = train_target_labels + [int(row[1]) for _ in range(len(files))]
        train_bone_type_labels = train_bone_type_labels + [bone_dic[bone_type] for _ in range(len(files))]

        if bone_type == 'XR_ELBOW' or bone_type == 'XR_WRIST':

          if int(row[1]) == 0:

            bone_dic_array[bone_type][0] = bone_dic_array[bone_type][0] + files
          elif int(row[1]) == 1:
            bone_dic_array[bone_type][1] = bone_dic_array[bone_type][1] + files



  create_tf_record_by_bone_type('XR_ELBOW', bone_dic_array)
  create_tf_record_by_bone_type('XR_WRIST', bone_dic_array)


def build_train_val_tfrecords(csv_data_dirs, main_tfrecords_dir, bone_dic):
    
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

  # build_train_tfrecords_by_bone_type(csv_data_dirs, main_tfrecords_dir, bone_dic)
  build_train_val_tfrecords(csv_data_dirs, main_tfrecords_dir, bone_dic)
  build_test_tfrecords(csv_data_dirs, main_tfrecords_dir, bone_dic)
