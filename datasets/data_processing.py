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

import os
import tensorflow as tf
import numpy as np

def read_tfrecords_nosaic_mnist(
    record_file_train, record_file_test, batch_size, 
    num_trainsubset=50000, shuffle_buffer_size=10000):
    """Reads TFRecord file and make parsed dataset tensors. 
    Returns train, validation and test dataset.
    Args:
        record_file_train: A string. 
            Path to training nosaic MNIST tfrecord file.
        record_file_test: A string. 
            Path to test nosaic MNIST tfrecord file.
        batch_size: An int.
        num_trainsubset: An int \in [1, 50000].
            Training on a subset of the training set is supported.
        shuffle_buffer_size: An int. 
            Larger size may lead to larger CPU memory consumption.
    Return:
        parsed_image_dataset_train: A dataset tensor.
        parsed_image_dataset_valid: A dataset tensor.
        parsed_image_dataset_test: A dataset tensor.
    Usage:
        # Training loop
        for i, feats in enumerate(parsed_image_dataset):
            video_batch = tf.io.decode_raw(feats['video'], tf.uint8)
            video_batch = tf.cast(video_batch, tf.float32)
            video_batch = tf.reshape(video_batch, (-1, 20, 28, 28, 1)) 
                # (B, T, H, W, C)
            label_batch = tf.cast(feats["label"], tf.int32) 
                # (B, )        
        # is equivalent to
        for i, feats in enumerate(parsed_image_dataset):
            video_batch, label_batch = decode_nosaic_mnist(feats)
    Remark:
        - drop_remainder=True for validation datasets 
            for simplicity of code.
    """
    assert 1 <= num_trainsubset <= 50000, "num_trainsubset = {}".format(
        num_trainsubset)
    assert batch_size <= num_trainsubset,\
        "batch_size={} must <= num_trainsubset={}".format(
            batch_size, num_trainsubset)
    
    def _parse_image_function(example_proto):
        return tf.io.parse_single_example(example_proto, {
                    'video': tf.io.FixedLenFeature([], tf.string),
                    'label': tf.io.FixedLenFeature([],tf.int64)
                    })

    raw_image_dataset = tf.data.TFRecordDataset(record_file_train)
    raw_image_dataset_test = tf.data.TFRecordDataset(record_file_test)
    parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
    parsed_image_dataset_test = raw_image_dataset_test.map(_parse_image_function)

    parsed_image_dataset_train = parsed_image_dataset.take(50000)
    parsed_image_dataset_valid = parsed_image_dataset.skip(50000)

    parsed_image_dataset_train = parsed_image_dataset_train.take(num_trainsubset)
    parsed_image_dataset_train = parsed_image_dataset_train.shuffle(shuffle_buffer_size)

    parsed_image_dataset_train = parsed_image_dataset_train.batch(
        batch_size, drop_remainder=True) ###
    parsed_image_dataset_valid = parsed_image_dataset_valid.batch(
        batch_size, drop_remainder=True) ###
    parsed_image_dataset_test = parsed_image_dataset_test.batch(
        batch_size, drop_remainder=True) ###

    return parsed_image_dataset_train, parsed_image_dataset_valid, parsed_image_dataset_test


def read_tfrecords_UCF101(
    record_file_tr, record_file_va, record_file_te, batch_size, flag_shuffle,
    shuffle_buffer_size=5000):
    """ Reads TFRecord file and create dataset tensors. 
        Returns train, validation and test datasets.
    Args:
        record_file_tr: A string. 
            Path to training tfrecord file.
        record_file_te: A string. 
            Path to test tfrecord file.
        record_file_te: A string. 
            Path to test tfrecord file.
        batch_size: An int.
        flag_shuffle: Shuffle train set or not.
            Validation and test sets are not shuffled anyway.
        shuffle_buffer_size: An int. 
            Larger size may lead to larger CPU memory consumption
            and a loading time per some training iterations to fill up
            the buffer (meaning IO bound).
    Returns:
        dataset_tr: tf.data.TFRecordDataset object.
        dataset_va: tf.data.TFRecordDataset object.
        dataset_te: tf.data.TFRecordDataset object.
    Usage:
        # Training loop
        for i, feats in enumerate(dataset_tr):
            videos = feats[0]
            videos = tf.reshape(videos, (-1, duration, feat_dim)) 
                # (batch, duration, feat dims)
            labels = feats[1]
                # (batch, )
    Remarks:
        - drop_remainder=False for validation and test sets.
    """
    
    def _parse_image_function(example_proto):
        features = tf.io.parse_single_example(example_proto, {
                    'video': tf.io.FixedLenFeature([], tf.string),
                    'label': tf.io.FixedLenFeature([], tf.int64)
                    })
        video = features["video"]
        label = features["label"]
        video = tf.io.decode_raw(video, tf.float32)
        label = tf.cast(label, tf.int32)

        return video, label

    dataset_tr = tf.data.TFRecordDataset(record_file_tr)
    dataset_va = tf.data.TFRecordDataset(record_file_va)
    dataset_te = tf.data.TFRecordDataset(record_file_te)
    dataset_tr = dataset_tr.map(_parse_image_function)
    dataset_va = dataset_va.map(_parse_image_function)
    dataset_te = dataset_te.map(_parse_image_function)

    if flag_shuffle:
        dataset_tr = dataset_tr.shuffle(shuffle_buffer_size)

    dataset_tr = dataset_tr.batch(batch_size, drop_remainder=True) ###
    dataset_va = dataset_va.batch(batch_size, drop_remainder=False)
    dataset_te = dataset_te.batch(batch_size, drop_remainder=False)

    return dataset_tr, dataset_va, dataset_te


def which_dataset_UCF101(duration, height, width, split):
    """
    Args:
        duration: An int.
        height: An int. Resolution of the original images.
        width: An int. Resolution of the original images.
        split: 1, 2, or 3. UCF101's train/test split num.
    Returns:
        Keywords for dictionary `config` (= load_yaml(Config Path)).
        Train/valid/test datasets for each;
        e.g., Path to training TFRecord file = config[key_tr].
    Remarks:
        T x H x W x C is ...
        UCF101 original: variable x 240 x 320 x 3
        UCF101-50-240-320: 50 x 240 x 320 x 3
        UCF101-50-256-256: 50 x 256 x 256 x 3
        UCF101-150-240-320: 150 x 240 x 320 x 3
        UCF101-150-256-256: 150 x 256 x 256 x 3
    """
    # Current support
    assert split == 1
    assert (height, width) == (240, 320) or (height, width) == (256, 256)
    assert np.any(duration == np.array([50, 150]))

    key_tr = "UCF101-{}-{}-{}-tr{}".format(duration, height, width, split)
    key_va = "UCF101-{}-{}-{}-va{}".format(duration, height, width, split)
    key_te = "UCF101-{}-{}-{}-te{}".format(duration, height, width, split)
    key_numtr = "numtr{}_UCF101-{}-{}-{}".format(split, duration, height, width)
    key_numva = "numva{}_UCF101-{}-{}-{}".format(split, duration, height, width)
    key_numte = "numte{}_UCF101-{}-{}-{}".format(split, duration, height, width)

    return key_tr, key_va, key_te, key_numtr, key_numva, key_numte


def which_dataset_HMDB51(duration, height, width, split):
    """
    Args:
        duration: An int.
        height: An int. Resolution of the original images.
        width: An int. Resolution of the original images.
        split: 1, 2, or 3. HMDB51's train/test split num.
    Returns:
        Keywords for dictionary `config` (= load_yaml(Config Path)).
        Train/valid/test datasets for each;
        e.g., Path to training TFRecord file = config[key_tr].
    Remarks:
        T x H x W x C is ...
        HMDB51 original: variable x 240 x about 320 x 3
        HMDB51-79-240-320: 79 x 240 x about 320 x 3
        HMDB51-200-240-320: 200 x 240 x about 320 x 3
    """
    # Current support
    assert split == 1
    assert (height, width) == (240, 320) # or (height, width) == (256, 256)
    assert np.any(duration == np.array([79, 200]))

    key_tr = "HMDB51-{}-{}-{}-tr{}".format(duration, height, width, split)
    key_va = "HMDB51-{}-{}-{}-va{}".format(duration, height, width, split)
    key_te = "HMDB51-{}-{}-{}-te{}".format(duration, height, width, split)
    key_numtr = "numtr{}_HMDB51-{}-{}-{}".format(split, duration, height, width)
    key_numva = "numva{}_HMDB51-{}-{}-{}".format(split, duration, height, width)
    key_numte = "numte{}_HMDB51-{}-{}-{}".format(split, duration, height, width)

    return key_tr, key_va, key_te, key_numtr, key_numva, key_numte
    

def decode_nosaic_mnist(features, duration, dtype_feat=tf.float32, dtype_label=tf.int32):
    """Decode TFRecords.
    Returns:
        video_batch: A Tensor with shape 
            (batch, duration, height, width, channel) 
            = (-1, 20, 28 ,28 ,1) 
            that represents a batch of videos. float32.
        label_batch: A Tensor with shape (batch,) 
            that represents a batch of labels. int32.
    Examle:
        parsed_image_dataset, _, _ = read_tfrecords_nosaic_mnist(
            path, batch size)
        for i, feats in enumerate(parsed_image_dataset):
            video_batch, label_batch = decode_nosaic_mnist(feats)
    """
    video_batch = tf.io.decode_raw(features['video'], tf.uint8)
    video_batch = tf.cast(video_batch, dtype_feat)
    video_batch = tf.reshape(video_batch, (-1, duration, 28, 28, 1)) 
        # (B, T, H, W, C)
    label_batch = tf.cast(features["label"], dtype_label) 
        # (B, )        

    return video_batch, label_batch


def decode_feat(features, duration, feat_dim, dtype_feat=tf.float32,
    dtype_label=tf.int32):
    """Decode TFRecords.
    Returns:
        video_batch: A Tensor with shape (batch, duration, feat dim)
            that represents a batch of frames of features.
        label_batch: A Tensor with shape (batch,) 
            that represents a batch of labels. int32.
    Usage:
        parsed_image_dataset, _, _ = read_tfrecords_nosaic_mnist(path, feat dim)
        for i, feats in enumerate(parsed_image_dataset):
            video_batch, label_batch = decode_nosaic_mnist(feats)
    """
    video_batch = tf.io.decode_raw(features['video'], dtype_feat)
    video_batch = tf.reshape(video_batch, (-1, duration, feat_dim)) # (B, T, D)
    label_batch = tf.cast(features["label"], dtype_label) # (B, )        

    return video_batch, label_batch


def binarize_labels_nosaic_mnist(labels):
    """Change labels like even (class 0) vs odd (class 1) numbers
    """
    labels = labels % 2
    return labels


def normalize_images_nosaic_mnist(images):
    images /= 127.5
    images -= 1
    return images


def reshape_for_featext(x, y, feat_dims):
    """(batch, duration) to (batch * duration,)"""
    x_shape = x.shape
    batch_size = x_shape[0]
    duration = x_shape[1]

    # To disentangle, tf.reshape(x, (batch, duration, feat_dims[0]...))
    x = tf.reshape(
        x, (-1, feat_dims[0], feat_dims[1], feat_dims[2]))

    y = tf.tile(y, [duration,])
    y = tf.reshape(y, (duration, batch_size))
    y = tf.transpose(y, [1,0])
    y = tf.reshape(y, (-1,))

    return x, y


def sequential_slice(x, y, order_sprt):
    """Slice, copy, and concat a batch to make a time-sliced, augumented batch.
    Effective batch size will be batch * (duration - order_sprt)).
    e.g., nosaic MNIST and 2nd-order SPRT: 
        effective batch size is (20-2)=18 times larger 
        than the original batch size.
    Args:
        x: A Tensor with shape 
            (batch, duration, feature dimension).
        y: A Tensor with shape (batch).
        order_sprt: An int. The order of SPRT.
    Returns:
        x_slice: A Tensor with shape 
            (batch*(duration-order_sprt), order_sprt+1, feat dim).
        y_slice: A Tensor with shape 
            (batch*(duration-order_sprt),).
    Remark:
        - y_slice may be a confusing name, because we copy and concatenate
          original y to obtain y_slice.
    """
    duration = x.shape[1]
    if duration < order_sprt + 1:
        raise ValueError(
        "order_sprt must be <= duration - 1."+\
        " Now order_sprt={}, duration={} .".format(
            order_sprt, duration))

    for i in range(duration - order_sprt):
        if i == 0:
            x_slice = x[:, i:i+order_sprt+1, :]
            y_slice = y
        else:
            x_slice = tf.concat([x_slice, x[:, i:i+order_sprt+1, :]],0)
            y_slice = tf.concat([y_slice, y], 0)

    return x_slice, y_slice


def sequential_concat(x_slice, y_slice, duration):
    """Opposite operation of sequential_slice. 
    x_slice's shape will change 
    from (batch * (duration - order_sprt), order_sprt + 1, feat dim )
    to  (batch, (duration - order_sprt), order_sprt + 1, feat dim).
    y changes accordingly.
    Args:
        x_slice: A Tensor with shape 
            (batch * (duration - order_sprt), order_sprt + 1, feat dim). 
            This is the output of 
            models.backbones_lstm.LSTMModel.__call__(inputs, training). 
        y_slice: A Tensor with shape (batch*(duration - order_sprt),).
        duration: An int. 20 for nosaic MNIST.
    Returns:
        x_cocnat: A Tensor with shape 
            (batch, (duration - order_sprt), order_sprt + 1, feat dim).
        y_concat: A Tensor with shape (batch).
    Remark:
        - y_concat may be a confusing name, because we slice 
          the input argument y_slice to get y_concat. Besides,
          y_concat is the same as y in `sequential_slice` function,
          in the first place...
    """
    x_shape = x_slice.shape
    order_sprt = int(x_shape[1] - 1)
    batch = int(x_shape[0] / (duration - order_sprt))
    feat_dim = x_shape[-1]

    # Cancat time-sliced, augumented batch
    x_concat = tf.reshape(
        x_slice, 
        (duration - order_sprt, batch, order_sprt + 1, feat_dim))
    x_concat = tf.transpose(x_concat, [1, 0, 2, 3]) 
        # (batch, duration - order_sprt, order_sprt + 1, feat_dim)
    y_concat = y_slice[:batch]

    return x_concat, y_concat