# MIT License

# Copyright (c) 2021 Taiki Miyagawa and Akinori F. Ebihara

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ==============================================================================

""" 
This script makes NMNIST-100f in the TFRecord format. 
Make sure to run this script twice 
to create both training and test datasets.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, time
import PIL
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

num_frames = 100 # = 100 for NMNIST-100f
num_rawpix = 15 # = 15 for NMNIST-100f
    # Num of raw-valued pixels in each frame.

############## USER DEFINED #####################
data_dir = "./tensorflow_datasets" 
    # MNIST will be downloaded to this directory
train_or_test = "test" # "train" or "test"
    # "train" (train dataset will be made) or "test" (test dataset will be made). 
    # Make sure to execute this script TWICE. 
    # One is to make the training dataset; the othrer is to make the test dataset.
record_file = "./nosaic_mnist-{0}f-{1}pix_{2}.tfrecords".\
    format(num_frames, num_rawpix, train_or_test)
    # TFR name
gpu = 1 # GPU number. If None, set_gpu_devices will not be executed.
#################################################

# Functions
def set_gpu_devices(gpu):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_visible_devices(physical_devices[gpu], 'GPU')
    tf.config.experimental.set_memory_growth(physical_devices[gpu], True)

def np_to_tfr_nosaic_mnist(x, y, writer):
    """Save a np.array to a tfrecord file. DO NOT FORGET writer.close().
    Args:
        x: data: np.ndarray, dtype=uint8
        y: label: int, dtype=int64
        writer: tf.io.TFRecordWriter object. Don't forget writer.close()
    """
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    # Make an Example object that has one record data
    example = tf.train.Example(features=tf.train.Features(feature={
        'video': _bytes_feature(x.tostring()),
        'label': _int64_feature(y)
        }))

    # Serialize the example object and make a TFRecord file
    writer.write(example.SerializeToString())

def fix_random_seed(flag_seed, seed=None):
    if flag_seed:
        np.random.seed(seed)
        tf.random.set_seed(seed)
        print("Numpy and TensorFlow's random seeds fixed: seed=" + str(seed))
    
    else:
        print("Random seed not fixed.")

if gpu is not None:
    set_gpu_devices(gpu) # the first GPU (numbered 0) will be used

# Fix seed
fix_random_seed(True, 7)

# Load MNIST as numpy array
############################
dstr, dsts = tfds.load(name="mnist", data_dir=data_dir, split=["train", "test"], batch_size=-1)
images_train = dstr["image"].numpy() # (60000, 28, 28, 1), np.uint8
images_test = dsts["image"].numpy()  # (60000,), np.int64
labels_train = dstr["label"].numpy() # (10000, 28, 28, 1), np.uint8
labels_test = dsts["label"].numpy()  # (10000,), np.int64

# train or test
if train_or_test == "train":
    images_make = images_train
    labels_make = labels_train
elif train_or_test == "test":
    images_make = images_test
    labels_make = labels_test
else:
    raise ValueError(train_or_test)

# Make training "OR" test data of nosaic MNIST
############################################################
if os.path.exists(record_file):
    raise ValueError("record_file exists. Remove or rename it.")
    
with tf.io.TFRecordWriter(record_file) as writer:
    cnt = 1
    for image, label in zip(images_make, labels_make):
        # Verbose
        if cnt % 1000 == 0:
            print("Iteration {}/{}".format(cnt, len(images_make)))
        cnt += 1

        # Reshape    
        image_reshape = np.reshape(image, (1, 28, 28))
        image_reshape = np.tile(image_reshape, (num_frames, 1, 1))
            # (num frames, 28, 28)

        # 1. Generate a sequence of random noise
        ########################################
        idx_rawpix = np.random.randint(0, 784, (num_frames, num_rawpix), dtype=np.int32)
            # (num frames, num rawpix)
        bool_rawpix = np.zeros((num_frames, 784), dtype=np.uint8)
            # (num frames, 784)
            
        for i, v in enumerate(idx_rawpix):
            # v.shape = (num rawpix,)
            bool_rawpix[i, v] += 1
            
        bool_rawpix = np.reshape(bool_rawpix, (num_frames, 28, 28))
            # (num frames, 28, 28)

        # 2. Masked images to a video
        ########################################
        masked_images = np.where(bool_rawpix, image_reshape, 255)
            # (28, 28)

        # Reshape
        video = np.reshape(masked_images, (num_frames, 28, 28, 1))
            # (num frames, 28, 28, 1)

        # 4. Save images
        ########################################
        label = np.int64(label) 
        video = np.uint8(video)
        np_to_tfr_nosaic_mnist(x=video, y=label, writer=writer)