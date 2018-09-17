#!/usr/bin/env python
from __future__ import division
import tensorflow as tf
import glob
import numpy as np
import os
import csv
from tfrecord import create_tf_record

def build_train_val_tfrecords(csv_data_dirs, main_tfrecords_dir):
    
  patient_csv_file = os.path.join(csv_data_dirs, 'train_labeled_studies.csv')
  val_csv_file = os.path.join(csv_data_dirs, 'val.csv')

  val_patient_files = []
  with open(val_csv_file , 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
      val_patient_files.append(row[0])

  train_files = []
  train_labels = []

  val_files = []
  val_labels = []

  with open(patient_csv_file , 'rb') as f:
    reader = csv.reader(f)
    for row in reader:

      if row[0] not in val_patient_files:
        files = glob.glob(os.path.join('/Users/andreas/Desktop/mura/',row[0]) + '/*.png')
        train_files = train_files + files 
        train_labels = train_labels + [int(row[1]) for _ in range(len(files))]
      else:
        files = glob.glob(os.path.join('/Users/andreas/Desktop/mura/',row[0]) + '/*.png')
        val_files = val_files + files 
        val_labels = val_labels + [int(row[1]) for _ in range(len(files))]
# 
  # create_tf_record(os.path.join(main_tfrecords_dir, 'train.tfrecords'), train_files, train_labels, True)
  create_tf_record(os.path.join(main_tfrecords_dir, 'val.tfrecords'), val_files, val_labels, True)

def build_test_tfrecords(csv_data_dirs, main_tfrecords_dir):

  patient_csv_file = os.path.join(csv_data_dirs, 'train_labeled_studies.csv')

  test_files = []
  test_labels = []

  with open(patient_csv_file , 'rb') as f:
    reader = csv.reader(f)
    for row in reader:

      files = glob.glob(os.path.join('/Users/andreas/Desktop/mura/',row[0]) + '/*.png')
      test_files = test_files + files 
      test_labels = test_labels + [int(row[1]) for _ in range(len(files))]

  create_tf_record(os.path.join(main_tfrecords_dir, 'test.tfrecords'), test_files, test_labels, True)

if __name__ == '__main__':

  csv_data_dirs = '/Users/andreas/Desktop/mura/MURA-v1.1/'
  main_tfrecords_dir = '/Users/andreas/Desktop/tfrecords/'

  build_train_val_tfrecords(csv_data_dirs, main_tfrecords_dir)
  # build_test_tfrecords(csv_data_dirs, main_tfrecords_dir, 'val')
