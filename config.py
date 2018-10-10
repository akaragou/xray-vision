import os
import tensorflow as tf

class XRAYconfig():
  def __init__(self, **kwargs):

    # directories for storing tfrecords, checkpoints etc.
    self.main_dir = '../'
    self.checkpoint_path = os.path.join(self.main_dir, 'checkpoints')
    self.summary_path = os.path.join(self.main_dir, 'summaries')
    self.results_path = os.path.join(self.main_dir, 'results')

    self.train_fn = os.path.join(self.main_dir, 'tfrecords/train.tfrecords')
    self.val_fn = os.path.join(self.main_dir, 'tfrecords/val.tfrecords')
    self.test_fn = os.path.join(self.main_dir, 'tfrecords/test.tfrecords')

    self.features = os.path.join(self.main_dir, 'features') 

    self.model_path = '/home/ubuntu/model_weights/resnet_v2_101.ckpt'
    self.restore_checkpoint = '/home/ubuntu/checkpoints/densenet_121/densenet_121_2018_10_06_19_57_56_26100.ckpt'

    self.output_shape = 2
    self.num_bones = 7
    self.restore = True
    self.optimizer = "adam"
    self.l2_reg = 0.00005
    self.batch_norm_decay = 0.999,
    self.batch_norm_epsilon = 0.001
    self.initial_learning_rate = 0.0001
    self.momentum = 0.9  # if optimizer is nestrov
    self.decay_steps = 5000 # number of steps before decaying the learning rate
    self.learning_rate_decay_factor = 0.1
    self.train_batch_size = 64
    self.val_batch_size = 64
    self.num_batches_to_validate_over = 3 # number of batches to validate over 32*100 = 3200
    self.validate_every_num_steps = 100 # perform a validation step
    self.num_train_epochs = 10000
    self.input_image_size = [256, 256, 3] # size of the input tf record image
    self.model_image_size = [224, 224, 3] # image dimesions that the model takes in

    # various options for altering input images during training and validation
    self.train_augmentations_dic = { 
                                  'rand_crop':True,
                                  'rand_flip_left_right':True,
                                  'rand_flip_top_bottom':False,
                                  'zoom':True,
                                  'shear':False,
                                  'shift':False,
                                  'brightness_contrast':False,
                                  'rand_rotate':True,
                                  'elastic_deformation':False,
                                  }

    self.val_augmentations_dic = { 
                                  'rand_crop':True,
                                  'rand_flip_left_right':True,
                                  'rand_flip_top_bottom':False,
                                  'zoom':True,
                                  'shear':False,
                                  'shift':False,
                                  'brightness_contrast':False,
                                  'rand_rotate':True,
                                  'elastic_deformation':False,
                                  }

    self.checkpoint_exclude_scopes = [
                                      "resnet_v2_101/block1/unit_1/bottleneck_v2/se_block",
                                      "resnet_v2_101/block1/unit_2/bottleneck_v2/se_block",
                                      "resnet_v2_101/block1/unit_3/bottleneck_v2/se_block",

                                      "resnet_v2_101/block2/unit_1/bottleneck_v2/se_block",
                                      "resnet_v2_101/block2/unit_2/bottleneck_v2/se_block",
                                      "resnet_v2_101/block2/unit_3/bottleneck_v2/se_block",
                                      "resnet_v2_101/block2/unit_4/bottleneck_v2/se_block",

                                      "resnet_v2_101/block3/unit_1/bottleneck_v2/se_block",
                                      "resnet_v2_101/block3/unit_2/bottleneck_v2/se_block",
                                      "resnet_v2_101/block3/unit_3/bottleneck_v2/se_block",
                                      "resnet_v2_101/block3/unit_4/bottleneck_v2/se_block",
                                      "resnet_v2_101/block3/unit_5/bottleneck_v2/se_block",
                                      "resnet_v2_101/block3/unit_6/bottleneck_v2/se_block",
                                      "resnet_v2_101/block3/unit_7/bottleneck_v2/se_block",
                                      "resnet_v2_101/block3/unit_8/bottleneck_v2/se_block",
                                      "resnet_v2_101/block3/unit_9/bottleneck_v2/se_block",
                                      "resnet_v2_101/block3/unit_10/bottleneck_v2/se_block",
                                      "resnet_v2_101/block3/unit_11/bottleneck_v2/se_block",
                                      "resnet_v2_101/block3/unit_12/bottleneck_v2/se_block",
                                      "resnet_v2_101/block3/unit_13/bottleneck_v2/se_block",
                                      "resnet_v2_101/block3/unit_14/bottleneck_v2/se_block",
                                      "resnet_v2_101/block3/unit_15/bottleneck_v2/se_block",
                                      "resnet_v2_101/block3/unit_16/bottleneck_v2/se_block",
                                      "resnet_v2_101/block3/unit_17/bottleneck_v2/se_block",
                                      "resnet_v2_101/block3/unit_18/bottleneck_v2/se_block",
                                      "resnet_v2_101/block3/unit_19/bottleneck_v2/se_block",
                                      "resnet_v2_101/block3/unit_20/bottleneck_v2/se_block",
                                      "resnet_v2_101/block3/unit_21/bottleneck_v2/se_block",
                                      "resnet_v2_101/block3/unit_22/bottleneck_v2/se_block",
                                      "resnet_v2_101/block3/unit_23/bottleneck_v2/se_block",

                                      "resnet_v2_101/block4/unit_1/bottleneck_v2/se_block",
                                      "resnet_v2_101/block4/unit_2/bottleneck_v2/se_block",
                                      "resnet_v2_101/block4/unit_3/bottleneck_v2/se_block",

                                      "resnet_v2_101/target_fc1",
                                      "resnet_v2_101/target_fc2",
                                      "resnet_v2_101/target_logits",
                                      "resnet_v2_101/target_spatial_squeeze",
                                     ]

  def get_checkpoint_filename(self, model_name, run_name):
    """ 
    Return filename for a checkpoint file. Ensure path exists
    Input: model_name - Name of the model
           run_name - Timestap of the training 
    Output: Full checkpoint filepath
    """
    pth = os.path.join(self.checkpoint_path, model_name)
    if not os.path.isdir(pth): os.makedirs(pth)
    return os.path.join(pth, run_name + '.ckpt')

  def get_summaries_path(self, model_name, run_name):
    """ 
    Return filename for a summaries file. Ensure path exists
    Input: model_name - Name of the model
           run_name - Timestap of the training 
    Output: Full summaries filepath
    """
    pth = os.path.join(self.summary_path, model_name)
    if not os.path.isdir(pth): os.makedirs(pth)
    return os.path.join(pth, run_name)

  def get_results_path(self, model_name, run_name):
    """ 
    Return filename for a results file. Ensure path exists
    Input: model_name - Name of the model
           run_name - Timestap of the training 
    Output: Full results filepath
    """
    pth = os.path.join(self.results_path, model_name, run_name)
    if not os.path.isdir(pth): os.makedirs(pth)
    return pth