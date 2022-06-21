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

import sys
import numpy as np
import tensorflow as tf
from datasets.data_processing import sequential_slice, sequential_concat
from utils.performance_metrics import calc_llrs, calc_oblivious_llrs, threshold_generator, cost_weight_generator


def multiplet_loss_func(logits_slice, labels_slice, 
    flag_prefactor=None, classwise_sample_sizes=None, beta=None):
    """ Multiplet loss for density estimation of time-series data.
        Cost-sensitive loss weighting is supported, 
        following the class-balanced loss:
        [Cui, Yin, et al. 
        "Class-balanced loss based on effective number of samples." 
        Proceedings of the IEEE/CVF Conference 
        on Computer Vision and Pattern Recognition. 2019.]
        (https://arxiv.org/abs/1901.05555).
    Args:
        model: A model.backbones_lstm.LSTMModel object. 
        logits_slice: An logit Tensor with shape 
            ((effective) batch size, order of SPRT + 1, num classes). 
            This is the output of LSTMModel.call(inputs, training).
        labels_slice: A label Tensor with shape ((effective) batch size,)
        flag_prefactor: A bool. Cost-sensitive or not.
        beta: A float larger than 0. Larger beta leads to 
            more discriminative weights.
            If beta is None, normal, non-cost-sensitive learning
            will be done. If beta = 1, then weights are simply 
            the inverse class frequencies (1 / N_k, 
            where N_k is the sample size of class k).
            If beta = -1, weights = [1,1,...,1] (len=num classes).
            This is useful for non-cost-sensitive learning.
        classwise_sample_sizes: A list of integers. 
            The length is equal to the number of classes.
    Returns:
        xent: A scalar Tensor. Sum of multiplet losses.
    Remarks:
        I'm not sure whether 
        weights *= effbs * (order_sprt + 1) / tf.reduce_sum(weights)
        is good or not (see the long Note below), 
        but it at least stabilize the scale variance of gradients.
    """
    if flag_prefactor:
        assert ((beta is not None) and (classwise_sample_sizes is not None))

    effbs, order_sprt, num_classes = logits_slice.shape
    order_sprt -= 1

    # Calc logits and reshape-with-copy labels
    logits = tf.transpose(logits_slice, [1, 0, 2])
    logits = tf.reshape(logits, [-1, num_classes])
    labels = tf.tile(labels_slice, [order_sprt + 1,])

    # Cost weight generation if necessary
    if flag_prefactor:
        weights = cost_weight_generator(classwise_sample_sizes, beta)
        weights = tf.gather(weights, indices=labels) 
        weights *= effbs * (order_sprt + 1) / tf.reduce_sum(weights) # good or bad for performance? dunno
            # shape = (effbs * (order_sprt + 1), ).
            # Note:
            # This additional normalization equalizes the scales of the multiplet loss 
            # with flag_prefactor = True and False. The reason follows.
            # Note that this additional normalization ensures that sum_{batch axis} weights = 1 
            # IN THE MULTIPLET LOSS, because tf.compat.v1.losses.sparse_softmax_cross_entropy
            # divides weights by `* effbs * (order_sprt + 1)` 
            # because of the reduce-mean in the batch axis.
            # (Note also that if weights = 1. (the `else` sentence below), 
            # then sum_{batch axis} weights = 1 is ensured in the multiplet loss).
    else:
        weights = 1.

    # Calc multiplet losses     
    xent = tf.compat.v1.losses.sparse_softmax_cross_entropy(
        labels=labels, logits=logits, weights=weights) 
        # A scalar averaged in a batch and sliding windows

    return xent


def margin_generator(llrs, labels_oh):
    """ Used in LLLR.
    Args:
        llrs: A Tensor with shape
            (batch, duration, num cls, num cls).
        labels_oh: A Tensor with shape
            (batch, 1, num cls, 1).
    Returns:
        random_margin: A Tensor with shape
            (batch, duration, num cls, num cls)
            - is negative for positive class row (\lambda_{k l} (k != y_i, l \in [num_classes]))
            - is posotive for negative class row (\lambda_{y_i k} (l \in [num_classes]))
            - All the margins share the same absolute value. 
    """
    labels_oh = 1 - (2 * labels_oh)
        # positive = -1, negative = 1
        # (batch, 1, num cls, 1)
    random_margin = threshold_generator(llrs, 1, "unirandom")
        # (1, duration, num cls, num cls)
        # positive values (diag = 0)
    random_margin = random_margin * labels_oh
        # (batch, duration, num cls, num cls)
        # negative for positive class row (\lambda_{k l} (k != y_i, l \in [num_classes]))
        # posotive for negative class row (\lambda_{y_i k} (l \in [num_classes])) 

    return random_margin


def LLLR(logits_concat, labels_concat, oblivious, version, flag_mgn, 
    flag_prefactor=None, flag_prior_ratio=None, classwise_sample_sizes=None, beta=None):
    """ LLLR for early multi-classification of time series.
        Cost-sensitive loss weighting is supported,
    Args:
        logits_concat: A logit Tensor with shape
            (batch, (duration - order_sprt), order_sprt + 1, num classes). 
            This is the output from 
            datasets.data_processing.sequential_concat(logit_slice, labels_slice)
        labels_concat: A label Tensor with shape (batch size,). 
            This is the output from 
            datasets.data_processing.sequential_concat(logit_slice, labels_slice).
        oblivious: A bool, whether to use TANDEMwO or not (= TANDEM).
        version: "A", "B", "C", "D", "E", or "F".
        flag_mgn: Use margin or not.
        flag_prior_ratio: A bool. Subtract the prior ratio term to the LLR or not.
        flag_prefactor: A bool. Add the prior ratio term to the LLR or not.
            Class-balanced-loss-like prior ratios supported.
        classwise_sample_sizes: A list of integers. 
            The length is equal to the number of classes.
        beta: A float larger than 0. Larger beta leads to 
            more discriminative weights.
            If beta is None, normal, non-cost-sensitive learning
            will be done. If beta = 1, then weights are simply 
            the inverse class frequencies (1 / N_k, 
            where N_k is the sample size of class k).
            If beta = -1, weights = [1,1,...,1] (len=num classes).
            This is useful for non-cost-sensitive learning.
    Return:
        llr_loss: A scalar Tensor that represents the log-likelihoood ratio loss.
    Remark:
        - version A: original (use all LLRs)
        - version B: simple LLLR (extract the positive class' raw)
        - version C: logistic LLLR (logistic loss instead of sigmoid)
        - version D: simple logistic LLLR (logitstic loss, extract the positive class' raw)
        - version E: LSEL (log-sum-exp loss)
        - version F: modLSEL
        - margin is generated uniformly-randomly.
    An example of assert checks for operation confirmation: 
        dummy = classwise_sample_sizes
        for _ in range(10):
            batch_size = 16
            duration = 50
            order_sprt = 20
            num_classes = 101
            beta = 1 # - 1e-2 # 1e-2 to 1e-5 or 1e-7
            logits = np.float32(np.random.rand(batch_size, duration, num_classes))
            labels = np.int32(np.random.randint(0, num_classes - 1, [batch_size]))
            logits_slice, labels_slice = sequential_slice(logits, labels, order_sprt)
            logits_concat, labels_concat = sequential_concat(logits_slice, labels_slice, duration)

            lllr1 = LLLR(logits_concat, labels_concat, oblivious=False, flag_mgn=False,
                version="E")
            lllr2 = LLLRv2(logits_concat, labels_concat, oblivious=False, flag_mgn=False,
               version="F", 
               flag_prefactor=False, 
               classwise_sample_sizes=dummy, beta=beta,
               flag_prior_ratio=True)
            assert np.float16(lllr1.numpy()) == np.float16(lllr2.numpy())

            lllr3 = LLLR(logits_concat, labels_concat, oblivious=True, flag_mgn=False, 
                version="E")
            lllr4 = LLLRv2(logits_concat, labels_concat, oblivious=True, flag_mgn=False,
                version="F",
                flag_prefactor=False, 
                classwise_sample_sizes=dummy, beta=beta,
                flag_prior_ratio=True)
            assert np.float16(lllr3.numpy()) == np.float16(lllr4.numpy())
    """
    # Preprocesses
    if flag_prior_ratio or flag_prefactor or version == "F":
        assert classwise_sample_sizes is not None

    shapes = logits_concat.shape
    #batch_size = shapes[0]
    order_sprt = shapes[2] - 1
    duration = shapes[1] + order_sprt
    num_classes = shapes[3]
    
    labels_oh = tf.one_hot(labels_concat, depth=num_classes, axis=1, dtype=tf.float32)
    labels_oh = tf.reshape(labels_oh, [-1, 1, num_classes, 1])
        # (batch, 1, num cls, 1)

    # Cost weight generation 
    if flag_prefactor or version == "F":
        weights_allcls = cost_weight_generator(classwise_sample_sizes, beta)
            # shape = (num classes, )
        weights = tf.gather(weights_allcls, indices=labels_concat)
            # shape = (batch, )

    else:
        weights_allcls = None
        weights = 1.        
        
    # Calc LLRs 
    if oblivious:
        llrs = calc_oblivious_llrs(logits_concat, 
            flag_prior_ratio, classwise_sample_sizes, beta) 
    else:
        llrs = calc_llrs(logits_concat,
            flag_prior_ratio, classwise_sample_sizes, beta) 
            # (batch, duration, num cls, num cls) 

    if version == "F":
        prior = tf.reshape(np.float32(weights_allcls), [1, 1, num_classes])
            # (1, 1, num cls, )
        llrs -= tf.math.log(tf.expand_dims(prior, axis=3) / tf.expand_dims(prior, axis=2))
            # (batch, duration, num cls, num cls)

    # Calc LLLR
    if flag_mgn:
        random_margin = margin_generator(llrs, labels_oh)
        llrs += random_margin
            # negative for positive class row (\lambda_{k l} (k != y_i, l \in [num_classes]))
            # posotive for negative class row (\lambda_{y_i k} (l \in [num_classes]))  

    if version == "A":
        lllr = tf.abs(labels_oh - tf.sigmoid(llrs))
            # (batch, duration, num cls, num cls)
        #lllr = 0.5 * (num_classes / (num_classes - 1.)) * tf.reduce_mean(lllr) 
        if flag_prefactor:
            lllr = tf.reduce_mean(lllr, axis=[1, 2, 3]) * weights
                # (batch,)
        lllr = 0.5 * (num_classes / (num_classes - 1.)) * tf.reduce_mean(lllr)

    elif version == "B":
        llrs = llrs * labels_oh
            # (batch, duration, num cls, num cls)
        llrs = tf.reduce_sum(llrs, axis=2)
            # (batch, duration, num cls)
        lllr = tf.abs(1. - tf.sigmoid(llrs))
        #lllr = (num_classes/ (num_classes - 1)) * tf.reduce_mean(lllr) 
        if flag_prefactor:
            lllr = tf.reduce_mean(lllr, axis=[1, 2]) * weights
                # (batch, )
        lllr = (num_classes/ (num_classes - 1)) * tf.reduce_mean(lllr)

    elif version == "C":
        labels_oh = tf.tile(labels_oh, [1, duration, 1, num_classes])
            # (batch, duration, num cls, num cls)
        lllr = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_oh, logits=llrs)
            # (batch, duration, num cls, num cls)
        #lllr = 0.5 * (num_classes/ (num_classes - 1)) * tf.reduce_mean(lllr) 
        if flag_prefactor:
            lllr = tf.reduce_mean(lllr, axis=[1, 2, 3]) * weights
                # (batch,)
        lllr = 0.5 * (num_classes/ (num_classes - 1)) * tf.reduce_mean(lllr) 

    elif version == "D":
        llrs = llrs * labels_oh
            # (batch, duration, num cls, num cls)
        llrs = tf.reduce_sum(llrs, axis=2)
            # (batch, duration, num cls)
        llrs = tf.transpose(llrs, [1, 0, 2])
            # (duration, batch, num cls)
        llrs = tf.reshape(llrs, [-1, num_classes])
            # (duration * batch, num cls)
            # A batch of llrs is repeated `duration` times.
        z = tf.ones_like(llrs, dtype=tf.float32)
        lllr = tf.nn.sigmoid_cross_entropy_with_logits(labels=z, logits=llrs)
            # (duration * batch, num cls)
        #lllr = (num_classes/ (num_classes - 1)) * tf.reduce_mean(lllr) 
        if flag_prefactor:
            weights = tf.tile(weights, [duration,])
                # (duration * batch,)
                # A batch is repeated `duration` times.
            lllr = tf.reduce_mean(lllr, axis=[1]) * weights
                # (duration * batch,)       
        lllr = (num_classes/ (num_classes - 1)) * tf.reduce_mean(lllr)        
        
    elif version == "E" or version == "F":
        llrs = llrs * labels_oh
            # (batch, duration, num cls, num cls)
        llrs = tf.reduce_sum(llrs, axis=2)
            # (batch, duration, num cls)
        llrs = tf.transpose(llrs, [1, 0, 2])
            # (duration, batch, num cls)
        llrs = tf.reshape(llrs, [-1, num_classes])
            # (duration * batch, num cls)
            # A batch of llrs is repeated `duration` times.
        minllr = tf.reduce_min(llrs, axis=1, keepdims=True)
            # (duration * batch, 1)
        llrs = llrs - minllr 
            # avoids over-/under-flow: 1st step
            # (duration * batch, num cls)
        expllrs = tf.reduce_sum(tf.math.exp(-llrs), axis=1) 
            # (duration * batch, )
            # Only the y_i-th rows survive.
        lllr = - minllr + tf.math.log(expllrs + 1e-12)
            # (duration * batch, 1)

        if flag_prefactor:
            weights = tf.tile(weights, [duration,])
                # (duration * batch,)
                # A batch is repeated `duration` times.
            lllr = lllr * weights
                # (duration * batch, 1)

        lllr = tf.reduce_mean(lllr)
            # avoids over-/under-flow: 2nd step
                
    else:
        raise ValueError(
            "version={} must be either of 'A', 'B', 'C', 'D', 'E'. or 'F'".\
                format(version))

    return lllr # scalar


def LLLR_ver2(logits_concat, labels_concat, oblivious, version, flag_mgn, 
    flag_prefactor=None, flag_prior_ratio=None, classwise_sample_sizes=None, beta=None):
    """ LLLR for early multi-classification of time series.
        Cost-sensitive loss weighting is supported,
    Args:
        logits_concat: A logit Tensor with shape
            (batch, (duration - order_sprt), order_sprt + 1, num classes). 
            This is the output from 
            datasets.data_processing.sequential_concat(logit_slice, labels_slice)
        labels_concat: A label Tensor with shape (batch size,). 
            This is the output from 
            datasets.data_processing.sequential_concat(logit_slice, labels_slice).
        oblivious: A bool, whether to use TANDEMwO or not (= TANDEM).
        version: "A", "B", "C", "D", "E", or "F".
        flag_mgn: Use margin or not.
        flag_prior_ratio: A bool. Subtract the prior ratio term to the LLR or not.
        flag_prefactor: A bool. Add the prior ratio term to the LLR or not.
            Class-balanced-loss-like prior ratios supported.
        classwise_sample_sizes: A list of integers. 
            The length is equal to the number of classes.
        beta: A float larger than 0. Larger beta leads to 
            more discriminative weights.
            If beta is None, normal, non-cost-sensitive learning
            will be done. If beta = 1, then weights are simply 
            the inverse class frequencies (1 / N_k, 
            where N_k is the sample size of class k).
            If beta = -1, weights = [1,1,...,1] (len=num classes).
            This is useful for non-cost-sensitive learning.
    Return:
        llr_loss: A scalar Tensor that represents the log-likelihoood ratio loss.
    Remark:
        - version A: original (use all LLRs)
        - version B: simple LLLR (extract the positive class' raw)
        - version C: logistic LLLR (logistic loss instead of sigmoid)
        - version D: simple logistic LLLR (logitstic loss, extract the positive class' raw)
        - version E: LSEL (log-sum-exp loss)
        - version F: modLSEL
        - margin is generated uniformly-randomly.
    An example of assert checks for operation confirmation: 
        dummy = classwise_sample_sizes
        for _ in range(10):
            batch_size = 16
            duration = 50
            order_sprt = 20
            num_classes = 101
            beta = 1 # - 1e-2 # 1e-2 to 1e-5 or 1e-7
            logits = np.float32(np.random.rand(batch_size, duration, num_classes))
            labels = np.int32(np.random.randint(0, num_classes - 1, [batch_size]))
            logits_slice, labels_slice = sequential_slice(logits, labels, order_sprt)
            logits_concat, labels_concat = sequential_concat(logits_slice, labels_slice, duration)

            lllr1 = LLLR(logits_concat, labels_concat, oblivious=False, flag_mgn=False,
                version="E")
            lllr2 = LLLRv2(logits_concat, labels_concat, oblivious=False, flag_mgn=False,
               version="F", 
               flag_prefactor=False, 
               classwise_sample_sizes=dummy, beta=beta,
               flag_prior_ratio=True)
            assert np.float16(lllr1.numpy()) == np.float16(lllr2.numpy())

            lllr3 = LLLR(logits_concat, labels_concat, oblivious=True, flag_mgn=False, 
                version="E")
            lllr4 = LLLRv2(logits_concat, labels_concat, oblivious=True, flag_mgn=False,
                version="F",
                flag_prefactor=False, 
                classwise_sample_sizes=dummy, beta=beta,
                flag_prior_ratio=True)
            assert np.float16(lllr3.numpy()) == np.float16(lllr4.numpy())
    """
    # Preprocesses
    if flag_prior_ratio or flag_prefactor or version == "F":
        assert classwise_sample_sizes is not None

    shapes = logits_concat.shape
    #batch_size = shapes[0]
    order_sprt = shapes[2] - 1
    duration = shapes[1] + order_sprt
    num_classes = shapes[3]
    
    labels_oh = tf.one_hot(labels_concat, depth=num_classes, axis=1, dtype=tf.float32)
    labels_oh = tf.reshape(labels_oh, [-1, 1, num_classes, 1])
        # (batch, 1, num cls, 1)

    # Cost weight generation 
    if flag_prefactor or version == "F":
        weights_allcls = cost_weight_generator(classwise_sample_sizes, beta)
            # shape = (num classes, )
        weights = tf.gather(weights_allcls, indices=labels_concat)
            # shape = (batch, )

    else:
        weights_allcls = None
        weights = 1.        
        
    # Calc LLRs 
    if oblivious:
        llrs = calc_oblivious_llrs(logits_concat, 
            flag_prior_ratio, classwise_sample_sizes, beta) 
    else:
        llrs = calc_llrs(logits_concat,
            flag_prior_ratio, classwise_sample_sizes, beta) 
            # (batch, duration, num cls, num cls) 
    llrs_output = llrs.numpy()

    if version == "F":
        prior = tf.reshape(np.float32(weights_allcls), [1, 1, num_classes])
            # (1, 1, num cls, )
        llrs -= tf.math.log(tf.expand_dims(prior, axis=3) / tf.expand_dims(prior, axis=2))
            # (batch, duration, num cls, num cls)

    # Calc LLLR
    if flag_mgn:
        random_margin = margin_generator(llrs, labels_oh)
        llrs += random_margin
            # negative for positive class row (\lambda_{k l} (k != y_i, l \in [num_classes]))
            # posotive for negative class row (\lambda_{y_i k} (l \in [num_classes]))  

    if version == "A":
        lllr = tf.abs(labels_oh - tf.sigmoid(llrs))
            # (batch, duration, num cls, num cls)
        #lllr = 0.5 * (num_classes / (num_classes - 1.)) * tf.reduce_mean(lllr) 
        if flag_prefactor:
            lllr = tf.reduce_mean(lllr, axis=[1, 2, 3]) * weights
                # (batch,)
        lllr = 0.5 * (num_classes / (num_classes - 1.)) * tf.reduce_mean(lllr)

    elif version == "B":
        llrs = llrs * labels_oh
            # (batch, duration, num cls, num cls)
        llrs = tf.reduce_sum(llrs, axis=2)
            # (batch, duration, num cls)
        lllr = tf.abs(1. - tf.sigmoid(llrs))
        #lllr = (num_classes/ (num_classes - 1)) * tf.reduce_mean(lllr) 
        if flag_prefactor:
            lllr = tf.reduce_mean(lllr, axis=[1, 2]) * weights
                # (batch, )
        lllr = (num_classes/ (num_classes - 1)) * tf.reduce_mean(lllr)

    elif version == "C":
        labels_oh = tf.tile(labels_oh, [1, duration, 1, num_classes])
            # (batch, duration, num cls, num cls)
        lllr = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_oh, logits=llrs)
            # (batch, duration, num cls, num cls)
        #lllr = 0.5 * (num_classes/ (num_classes - 1)) * tf.reduce_mean(lllr) 
        if flag_prefactor:
            lllr = tf.reduce_mean(lllr, axis=[1, 2, 3]) * weights
                # (batch,)
        lllr = 0.5 * (num_classes/ (num_classes - 1)) * tf.reduce_mean(lllr) 

    elif version == "D":
        llrs = llrs * labels_oh
            # (batch, duration, num cls, num cls)
        llrs = tf.reduce_sum(llrs, axis=2)
            # (batch, duration, num cls)
        llrs = tf.transpose(llrs, [1, 0, 2])
            # (duration, batch, num cls)
        llrs = tf.reshape(llrs, [-1, num_classes])
            # (duration * batch, num cls)
            # A batch of llrs is repeated `duration` times.
        z = tf.ones_like(llrs, dtype=tf.float32)
        lllr = tf.nn.sigmoid_cross_entropy_with_logits(labels=z, logits=llrs)
            # (duration * batch, num cls)
        #lllr = (num_classes/ (num_classes - 1)) * tf.reduce_mean(lllr) 
        if flag_prefactor:
            weights = tf.tile(weights, [duration,])
                # (duration * batch,)
                # A batch is repeated `duration` times.
            lllr = tf.reduce_mean(lllr, axis=[1]) * weights
                # (duration * batch,)       
        lllr = (num_classes/ (num_classes - 1)) * tf.reduce_mean(lllr)        
        
    elif version == "E" or version == "F":
        llrs = llrs * labels_oh
            # (batch, duration, num cls, num cls)
        llrs = tf.reduce_sum(llrs, axis=2)
            # (batch, duration, num cls)
        llrs = tf.transpose(llrs, [1, 0, 2])
            # (duration, batch, num cls)
        llrs = tf.reshape(llrs, [-1, num_classes])
            # (duration * batch, num cls)
            # A batch of llrs is repeated `duration` times.
        minllr = tf.reduce_min(llrs, axis=1, keepdims=True)
            # (duration * batch, 1)
        llrs = llrs - minllr 
            # avoids over-/under-flow: 1st step
            # (duration * batch, num cls)
        expllrs = tf.reduce_sum(tf.math.exp(-llrs), axis=1) 
            # (duration * batch, )
            # Only the y_i-th rows survive.
        lllr = - minllr + tf.math.log(expllrs + 1e-12)
            # (duration * batch, 1)

        if flag_prefactor:
            weights = tf.tile(weights, [duration,])
                # (duration * batch,)
                # A batch is repeated `duration` times.
            lllr = lllr * weights
                # (duration * batch, 1)

        lllr = tf.reduce_mean(lllr)
            # avoids over-/under-flow: 2nd step
                
    else:
        raise ValueError(
            "version={} must be either of 'A', 'B', 'C', 'D', 'E'. or 'F'".\
                format(version))

    return lllr, llrs_output # scalar

class LossDRE():
    def __init__(self, loss_type, duration, num_classes, oblivious,
        multLam=None, flag_prefactor=None, flag_prior_ratio=None,
        classwise_sample_sizes=None, beta=None):
        """
        Args:
            loss_type: A string. Current support is listed in self.current_support.
            duration: An int. Length of incoming sequences.
            num_classes: An int. Number of classes.
            oblivious: A bool, whether to use TANDEMwO or not (= TANDEM).
            multLam: A float or None. The prefactor of the constraint term of BARR. 
                If loss_type is "BARR", multLam must not be None.
            flag_prior_ratio: A bool. Subtract the prior ratio term to the LLR or not.
            flag_prefactor: A bool. Add the prior ratio term to the LLR or not.
                Class-balanced-loss-like prior ratios supported.
            classwise_sample_sizes: A list of integers. 
                The length is equal to the number of classes.
            beta: A float larger than 0. Larger beta leads to 
                more discriminative weights.
                If beta is None, normal, non-cost-sensitive learning
                will be done. If beta = 1, then weights are simply 
                the inverse class frequencies (1 / N_k, 
                where N_k is the sample size of class k).
                If beta = -1, weights = [1,1,...,1] (len=num classes).
                This is useful for non-cost-sensitive learning.
        """
        # Initialization
        self.current_support = [
            "LSIF",
            "LSIFwC", 
            "DSKL", 
            "BARR",
            "LLLR",
            "RuLSIF",
            "Logistic",
            "LSEL",
            "NGA-LSEL"]
        self.loss_type = loss_type
        self.duration = duration
        self.num_classes = num_classes
        self.oblivious = oblivious
        self.multLam = multLam
        self.flag_prefactor = flag_prefactor
        self.flag_prior_ratio = flag_prior_ratio
        self.classwise_sample_sizes = classwise_sample_sizes
        self.beta = beta

        # Assert
        assert np.any(loss_type == np.array(self.current_support))
        if loss_type == "BARR":
            assert multLam is not None
        elif loss_type == "LSIFwC":
            assert multLam is not None
        else:
            assert multLam is None

        if loss_type == "RuLSIF":
            raise ValueError("loss_type = 'RuLSIF' is not currently supported.")

        if self.flag_prior_ratio or self.flag_prefactor:
            assert self.classwise_sample_sizes is not None

        # Weights for later use
        if self.flag_prefactor:
            self.weights_allcls = cost_weight_generator(self.classwise_sample_sizes, self.beta)
                # shape = (num classes, )


    def LSIF(self, llrs, labels_oh, weights):
        """
        LSIF is unbounded and can be - infinity.
        Args:
            llrs: A float Tensor with shape (batch, duration, num classes, num classes).
                LLR matrices.
            labels_oh: A float Tensor with shape (batch, num classes). 
                One-hot labels.
            weights: 1. or a float Tensor with shape (batch, ).
        Returns:
            loss: A scalar Tensor. LSIF objective.
            loss1: A scalar Tensor. The first term.
            loss2: A scalar Tensor. The second term (w/o the prefactor if any).
        """
        weights = tf.reshape(weights, (-1, 1, 1, 1))
            # (batch, 1, 1, 1)
        llrs += tf.math.log(weights + 1e-12)
            # (batch, duration, num cls, num cls)
        llrs = tf.clip_by_value(llrs, -35, 35.) 
            # loss is barely non-inf. (but llrs for SPRT may be ~200)
            # (batch, duration, num cls, num cls)

        labels1 = tf.reshape(labels_oh, (-1, 1, 1, self.num_classes))
            # (batch, 1, 1, num classes)
        scores1 = 2 * tf.reduce_sum(llrs * labels1, 3)
            # (batch, duration, num cls)
        scores1 = tf.exp(scores1)
            # (batch, duration, num cls)

        labels2 = tf.reshape(labels_oh, (-1, 1, self.num_classes, 1))
            # (batch, 1, num cls, 1)
        scores2 = tf.reduce_sum(llrs * labels2, 2)
            # (batch, duration, num cls)
        scores2 = - 2 * tf.exp(scores2)
            # (batch, duration, num cls)
        
        loss = tf.reduce_mean(scores1 + scores2) 
        loss1 = tf.reduce_mean(scores1)
        loss2 = tf.reduce_mean(scores2)

        return loss, loss1, loss2

    def LSIFwC(self, llrs, labels_oh, weights):
        """
        LSIFwC is bounded but can be negative.
        Args:
            llrs: A float Tensor with shape (batch, duration, num classes, num classes).
                LLR matrices.
            labels_oh: A float Tensor with shape (batch, num classes). 
                One-hot labels.
            weights: 1. or a float Tensor with shape (batch, ).
        Returns:
            loss: A scalar Tensor. LSIF objective.
            loss1: A scalar Tensor. The first term.
            loss2: A scalar Tensor. The second term (w/o the prefactor if any).
        """
        weights = tf.reshape(weights, (-1, 1, 1, 1))
            # (batch, 1, 1, 1)
        llrs += tf.math.log(weights + 1e-12)
            # (batch, duration, num cls, num cls)
        llrs = tf.clip_by_value(llrs, -35, 35.) 
            # loss is barely non-inf. (but llrs for SPRT may be ~200)
            # (batch, duration, num cls, num cls)

        labels1 = tf.reshape(labels_oh, (-1, 1, 1, self.num_classes))
            # (batch, 1, 1, num classes)
        scores1 = 2 * tf.reduce_sum(llrs * labels1, 3)
            # (batch, duration, num cls)
        scores1 = tf.exp(scores1)
            # (batch, duration, num cls)

        labels2 = tf.reshape(labels_oh, (-1, 1, self.num_classes, 1))
            # (batch, 1, num cls, 1)
        scores2 = tf.reduce_sum(llrs * labels2, 2)
            # (batch, duration, num cls)
        scores2 = - 2 * tf.exp(scores2)
            # (batch, duration, num cls)

        scores3 = llrs * labels1
            # (batch, duration, num cls, num cls)
        scores3 = tf.exp(scores3)
            # (batch, duration, num cls, num cls)
        scores3 = scores3 - weights 
            # corresponds to "-1" in | 1/n sum_i r_i - 1 | in BARR.
            # (batch, duration, num cls, num cls)
        scores3 = scores3 * labels1
            # (batch, duration, num cls, num cls)
        scores3 = tf.reduce_sum(scores3, 0)
            # (duration, num cls, num cls)
        scores3 = tf.abs(scores3)
            # (duration, num cls, num cls)
        scores3 = tf.reduce_sum(scores3, 2)
            # (duration, num cls)
        scores3 = tf.reduce_mean(scores3)

        loss = tf.reduce_mean(scores1 + scores2) + self.multLam * scores3
        loss1 = tf.reduce_mean(scores1)
        loss2 = tf.reduce_mean(scores2)

        return loss, loss1, loss2

    def DSKL(self, llrs, labels_oh, weights):
        """
        DSKL is unbounded and can be - infinity.
        Args:
            llrs: A float Tensor with shape (batch, duration, num classes, num classes).
                LLR matrices.
            labels_oh: A float Tensor with shape (batch, num classes). 
                One-hot labels.
            weights: 1. or a float Tensor with shape (batch, ).
        Returns:
            loss: A scalar Tensor. DSKL objective.
            loss1: A scalar Tensor. The first term.
            loss2: A scalar Tensor. The second term (w/o the prefactor if any).
        """
        weights = tf.reshape(weights, (-1, 1, 1))
            # (batch, 1, 1)

        labels1 = tf.reshape(labels_oh, (-1, 1, self.num_classes, 1))
            # (batch, 1, num classes, 1)
        scores1 = tf.reduce_sum(llrs * labels1, 2)
            # (batch, duration, num cls)
        scores1 = - weights * scores1
            # (batch, duration, num cls)

        labels2 = tf.reshape(labels_oh, (-1, 1, 1, self.num_classes))
            # (batch, 1, 1, num cls)
        scores2 = tf.reduce_sum(llrs * labels2, 3)
            # (batch, duration, num cls)
        scores2 = weights * scores2
            # (batch, duration, num cls)

        loss = tf.reduce_mean(scores1 + scores2)
        loss1 = tf.reduce_mean(scores1)
        loss2 = tf.reduce_mean(scores2)

        return loss, loss1, loss2

    def BARR(self, llrs, labels_oh, weights):
        """
        BARR is bounded but can be negative.
        Args:
            llrs: A float Tensor with shape (batch, duration, num classes, num classes).
                LLR matrices.
            labels_oh: A float Tensor with shape (batch, num classes). 
                One-hot labels.
            weights: 1. or a float Tensor with shape (batch, ).
        Returns:
            loss: A scalar Tensor. BARR objective.
            loss1: A scalar Tensor. The first term.
            loss2: A scalar Tensor. The second term (w/o the prefactor if any).
        """
        weights = tf.reshape(weights, (-1, 1, 1))
            # (batch, 1, 1)

        labels1 = tf.reshape(labels_oh, (-1, 1, self.num_classes, 1))
            # (batch, 1, num classes, 1)
        scores1 = tf.reduce_sum(llrs * labels1, 2)
            # (batch, duration, num cls)
        scores1 = - weights * scores1
            # (batch, duration, num cls)
        scores1 = tf.reduce_mean(scores1)

        labels2 = tf.reshape(labels_oh, (-1, 1, 1, self.num_classes))
            # (batch, 1, 1, num cls)
        weights = tf.expand_dims(weights, 3)
            # (batch, 1, 1, 1)
        scores2 = llrs + tf.math.log(weights + 1e-12)
            # (batch, duration , num cls, num cls) 
        scores2 = scores2 * labels2
            # (batch, duration, num cls, num cls)
        scores2 = tf.clip_by_value(scores2, -35., 35.)
            # (batch, duration, num cls, num cls)
        scores2 = tf.exp(scores2)
            # (batch, duration, num cls, num cls)
        scores2 = scores2 - weights
            # (batch, duration, num cls, num cls)
        scores2 = scores2 * labels2
            # (batch, duration, num cls, num cls)
        scores2 = tf.reduce_sum(scores2, 0)
            # (duration, num cls, num cls)
        scores2 = tf.abs(scores2)
            # (duration, num cls, num cls)
        scores2 = tf.reduce_sum(scores2, 2)
            # (duration, num cls)
        scores2 = tf.reduce_mean(scores2)
        
        loss = scores1 + self.multLam * scores2 
            # is very likely to cause the loss of trailing digits, 
            # if scores2 = O(10).
            # And maybe is in the backprop...

        return loss, scores1, scores2

    def LLLR(self, llrs, labels_oh, weights):
        """
        LLLR is bounded and positive.
        Args:
            llrs: A float Tensor with shape (batch, duration, num classes, num classes).
                LLR matrices.
            labels_oh: A float Tensor with shape (batch, num classes). 
                One-hot labels.
            weights: 1. or a float Tensor with shape (batch, ).
        Returns:
            loss: A scalar Tensor. LLLR objective.
            loss1: A scalar Tensor. The first term.
            loss2: A scalar Tensor. The second term (w/o the prefactor if any).
        """
        weights = tf.reshape(weights, (-1, 1, 1))
            # (batch, 1, 1)

        factor1 = tf.tile(1. / (weights + 1e-12), [1, self.num_classes, 1])
        factor2 = tf.tile(1. / (weights + 1e-12), [1, 1, self.num_classes])
        factor = factor1 + factor2
        factor = 1. / (factor + 1e-12)
            # (batch, num cls, num cls)
        factor = tf.expand_dims(factor, 1)
            # (batch, 1, num cls, num cls)

        labels1 = tf.reshape(labels_oh, (-1, 1, 1, self.num_classes))
            # (batch, 1, 1, num classes)
        scores1 = tf.sigmoid(llrs) * factor
            # (batch, duration, num cls, num cls)
        scores1 = scores1 * labels1 * factor
            # (batch, duration, num cls, num cls)
        scores1 = tf.reduce_mean(scores1)

        labels2 = tf.reshape(labels_oh, (-1, 1, self.num_classes, 1))
            # (batch, 1, num cls, 1)
        scores2 = tf.abs(1 - tf.sigmoid(llrs))
            # (batch, duration, num cls, num cls)
        scores2 = scores2 * labels2 * factor
            # (batch, duration, num cls, num cls)
        scores2 = tf.reduce_mean(scores2)
        
        loss = scores1 + scores2

        return loss, scores1, scores2

    def RuLSIF(self):
        raise ValueError("loss_type = 'RuLSIF' is not currently supported.")

    def Logistic(self, llrs, labels_oh, weights):
        """ The logistic loss.
        Args:
            llrs: A float Tensor with shape (batch, duration, num classes, num classes).
                LLR matrices.
            labels_oh: A float Tensor with shape (batch, num classes). 
                One-hot labels.
            weights: 1. or a float Tensor with shape (batch, ).
        Returns:
            loss: A scalar Tensor. Logistic loss' objective.
            loss1: 0. 
            loss2: 0. 
        """
        labels_oh = tf.reshape(labels_oh, [-1, 1, self.num_classes, 1])
            # (batch, 1, num cls, 1)
        llrs = llrs * labels_oh
            # (batch, duration, num cls, num cls)
        llrs = tf.reduce_sum(llrs, axis=2)
            # (batch, duration, num cls)
        llrs = tf.transpose(llrs, [1, 0, 2])
            # (duration, batch, num cls)
        llrs = tf.reshape(llrs, [-1, self.num_classes])
            # (duration * batch, num cls)
            # A batch of llrs is repeated `duration` times.
        z = tf.ones_like(llrs, dtype=tf.float32)
        lllr = tf.nn.sigmoid_cross_entropy_with_logits(labels=z, logits=llrs)
            # (duration * batch, num cls)

        if self.flag_prefactor:
            weights = tf.tile(weights, [self.duration,])
                # (duration * batch,)
                # A batch is repeated `duration` times.
            lllr = tf.reduce_mean(lllr, axis=[1]) * weights
                # (duration * batch,)       

        loss = (self.num_classes/ (self.num_classes - 1)) * tf.reduce_mean(lllr)   
        loss1 = 0.
        loss2 = 0.

        return loss, loss1, loss2

    def LSEL(self, llrs, labels_oh, weights):
        """ The log-sum-exp loss (LSEL). Positive, consistent, numerically stable, and efficient.
        Args:
            llrs: A float Tensor with shape (batch, duration, num classes, num classes).
                LLR matrices.
            labels_oh: A float Tensor with shape (batch, num classes). 
                One-hot labels.
            weights: 1. or a float Tensor with shape (batch, ).
        Returns:
            loss: A scalar Tensor. LSEL objective.
            loss1: 0. 
            loss2: 0. 
        """
        labels_oh = tf.reshape(labels_oh, [-1, 1, self.num_classes, 1])
            # (batch, 1, num cls, 1)
        llrs = llrs * labels_oh
            # (batch, duration, num cls, num cls)
        llrs = tf.reduce_sum(llrs, axis=2)
            # (batch, duration, num cls)
        llrs = tf.transpose(llrs, [1, 0, 2])
            # (duration, batch, num cls)
        llrs = tf.reshape(llrs, [-1, self.num_classes])
            # (duration * batch, num cls)
            # A batch of llrs is repeated `duration` times.
        minllr = tf.reduce_min(llrs, axis=1, keepdims=True)
            # (duration * batch, 1)
        llrs = llrs - minllr 
            # avoids over-/under-flow: 1st step
            # (duration * batch, num cls)
        expllrs = tf.reduce_sum(tf.math.exp(-llrs), axis=1) 
            # (duration * batch, )
            # Only the y_i-th rows survive.
        lllr = - minllr + tf.math.log(expllrs + 1e-12)
            # (duration * batch, 1)

        if self.flag_prefactor:
            weights = tf.tile(weights, [self.duration,])
                # (duration * batch,)
                # A batch is repeated `duration` times.
            lllr = lllr * weights
                # (duration * batch, 1)

        loss = tf.reduce_mean(lllr)
            # avoids over-/under-flow: 2nd step
        loss1 = 0.
        loss2 = 0.

        return loss, loss1, loss2

    def NGALSEL(self, llrs, labels_oh, weights):
        """ Non-guess-averse log-sum-exp loss (NGA-LSEL). Positive, numerically stable.
        Args:
            llrs: A float Tensor with shape (batch, duration, num classes, num classes).
                LLR matrices.
            labels_oh: A float Tensor with shape (batch, num classes). 
                One-hot labels.
            weights: 1. or a float Tensor with shape (batch, ).
        Returns:
            loss: A scalar Tensor. NGA-LSEL objective.
            loss1: 0. 
            loss2: 0. 
        """
        labels_oh = 1. - labels_oh
        labels_oh = tf.reshape(labels_oh, [-1, 1, self.num_classes, 1])
            # (batch, 1, num cls, 1)
        llrs = llrs * labels_oh
            # (batch, duration, num cls, num cls)
        llrs = tf.reduce_sum(llrs, axis=2)
            # (batch, duration, num cls)
        llrs = tf.transpose(llrs, [1, 0, 2])
            # (duration, batch, num cls)
        llrs = tf.reshape(llrs, [-1, self.num_classes])
            # (duration * batch, num cls)
            # A batch of llrs is repeated `duration` times.
        maxllr = tf.reduce_max(llrs, axis=1, keepdims=True)
            # (duration * batch, 1)
        llrs = llrs - maxllr 
            # avoids over-/under-flow: 1st step
            # (duration * batch, num cls)
        expllrs = tf.reduce_sum(tf.math.exp(llrs), axis=1) 
            # (duration * batch, )
            # Only the y_i-th rows survive.
        lllr =  maxllr + tf.math.log(expllrs + 1e-12)
            # (duration * batch, 1)

        if self.flag_prefactor:
            weights = tf.tile(weights, [self.duration,])
                # (duration * batch,)
                # A batch is repeated `duration` times.
            lllr = lllr * weights
                # (duration * batch, 1)

        loss = tf.reduce_mean(lllr)
            # avoids over-/under-flow: 2nd step
        loss1 = 0.
        loss2 = 0.

        return loss, loss1, loss2

    def __call__(self, logits_concat, labels_concat):
        """
        Args:
            logits_concat: A logit Tensor with shape
                (batch, (duration - order_sprt), order_sprt + 1, num classes). 
                This is the output from 
                datasets.data_processing.sequential_concat(logit_slice, labels_slice)
            labels_concat: A label Tensor with shape (batch size,). 
                This is the output from 
                datasets.data_processing.sequential_concat(logit_slice, labels_slice).
        Returns:
            loss: A scalar Tensor.
            loss1: A scalar Tensor. The fist term of the loss in use.
                For logging purposes.
            loss2: A scalar Tensor. The second term of the loss in use.
                For logging purposes.
        """
        shapes = logits_concat.shape
        num_classes = shapes[3]
        
        # Calc cost weights 
        ###################################################
        if self.flag_prefactor:
            weights = tf.gather(self.weights_allcls, indices=labels_concat)
                # shape = (batch, )

        else:
            weights = 1.        
            
        # Calc LLRs
        ###################################################
        if self.oblivious:
            llrs = calc_oblivious_llrs(logits_concat, 
                self.flag_prior_ratio, self.classwise_sample_sizes, self.beta) 
        else:
            llrs = calc_llrs(logits_concat,
                self.flag_prior_ratio, self.classwise_sample_sizes, self.beta) 
                # (batch, duration, num cls, num cls) 

        # Calc loss function
        ##################################################
        labels_oh = tf.one_hot(labels_concat, depth=num_classes, axis=1, dtype=tf.float32)
            # (batch, num cls)

        if self.loss_type == "LSIF":
            loss, loss1, loss2 = self.LSIF(llrs, labels_oh, weights)

        elif self.loss_type == "LSIFwC":
            loss, loss1, loss2 = self.LSIFwC(llrs, labels_oh, weights)

        elif self.loss_type == "DSKL":
            loss, loss1, loss2 = self.DSKL(llrs, labels_oh, weights)

        elif self.loss_type == "BARR":
            loss, loss1, loss2 = self.BARR(llrs, labels_oh, weights)

        elif self.loss_type == "LLLR":
            loss, loss1, loss2 = self.LLLR(llrs, labels_oh, weights)

        elif self.loss_type == "RuLSIF":
            raise ValueError("loss_type = 'RuLSIF' is not currently supported.")

        elif self.loss_type == "Logistic":
            loss, loss1, loss2 = self.Logistic(llrs, labels_oh, weights)

        elif self.loss_type == "LSEL":
            loss, loss1, loss2 = self.LSEL(llrs, labels_oh, weights)

        elif self.loss_type == "NGA-LSEL":
            loss, loss1, loss2 = self.NGALSEL(llrs, labels_oh, weights)

        else:
            raise ValueError(
                "loss_type not allowed. Got {}, but current support = {}".\
                format(self.loss_type, self.current_support))

        return loss, loss1, loss2


def get_gradient_DRE(model, x, y, training, order_sprt, duration, 
    flag_wd, calc_grad, param_wd, loss_function):
    """Calculate loss and/or gradients.
    Args:
        model: A tf.keras.Model object.
        x: A Tensor. A batch of time-series input data 
            without sequential_slice and sequential_concat.
        y: A Tensor. A batch of labels 
            without sequential_slice and sequential_concat
            = labels_concat.
        training: A boolean. Training flag.
        order_sprt: An int. The order of the SPRT.
        duration: An int. Num of frames in a sequence.
        param_wd: A float. Loss weight.
        flag_wd: A boolean. Weight decay or not.
        loss_function: A LossDRE instance.
    Returns:
        gradients: A Tensor or None.
        losses: A list of loss Tensors; namely,
            total_loss: A scalar Tensor or 0 if not calc_grad. 
                The weighted total loss.
            loss_dre: A scalar Tensor.
            loss1: A scalar Tensor. First term contribution.
            loss2: A scalar Tensor. Second term contribution.
            wd_reg: A scalar Tensor.
        logits_concat: A logit Tensor with shape 
            (batch, (duration - order_sprt), order_sprt + 1, num classes). 
            This is the output of 
            datasets.data_processing.sequential_concat(logit_slice, labels_slice).
    Remarks:
        All the losses below will be calculated if calc_grad is False 
        to write logs for TensorBoard.
            total_loss 
            loss_dre
    """
    x_slice, y_slice = sequential_slice(x, y, order_sprt)

    # For training
    if calc_grad:
        with tf.GradientTape() as tape:
            logits_slice = model(x_slice, training)
            logits_concat, labels_concat = sequential_concat(
                logits_slice, y_slice, duration)
            total_loss = 0.

            # loss_dre
            loss_dre, loss1, loss2 = loss_function(logits_concat, labels_concat)
            total_loss += loss_dre

            # wd_reg
            wd_reg = 0.
            if flag_wd:
                for variables in model.trainable_variables:
                    wd_reg += tf.nn.l2_loss(variables)
                    total_loss += param_wd * wd_reg

        gradients = tape.gradient(total_loss, model.trainable_variables)
        losses = [total_loss, loss_dre, loss1, loss2, wd_reg]

    # For validation and test
    else: 
        logits_slice = model(x_slice, training)
        logits_concat, labels_concat = sequential_concat(
            logits_slice, y_slice, duration)
        total_loss = 0.

        # loss_dre
        loss_dre, loss1, loss2 = loss_function(logits_concat, labels_concat)
        total_loss += loss_dre

        # wd_reg
        wd_reg = 0.
        # for variables in model.trainable_variables:
        #     wd_reg += tf.nn.l2_loss(variables)

        gradients = None
        losses = [0., loss_dre, loss1, loss2, wd_reg]

    return gradients, losses, logits_concat


def get_gradient_lstm(model, x, y, training, order_sprt, duration, 
    oblivious, version, flag_wd, flag_mgn, calc_grad, 
    param_multiplet_loss, param_llr_loss, param_wd,
    flag_prefactor=None, classwise_sample_sizes=None, beta=None,
    flag_prior_ratio=None):
    """Calculate loss and/or gradients.
    Args:
        model: A tf.keras.Model object.
        x: A Tensor. A batch of time-series input data 
            without sequential_slice and sequential_concat.
        y: A Tensor. A batch of labels 
            without sequential_slice and sequential_concat.
        training: A boolean. Training flag.
        order_sprt: An int. The order of the SPRT.
        duration: An int. Num of frames in a sequence.
        oblivious: A bool. TANDEMwO or normal TANDEM.
        version: A string, "A", "B", "C", or "D".
        param_multiplet_loss: A float. Loss weight.
        param_llr_loss: A float. Loss weight.
        param_wd: A float. Loss weight.
        flag_wd: A boolean. Weight decay or not.
        flag_mgn: A boolean. Use margin in LLLR or not.
        flag_prior_ratio: A bool. Add the prior ratio term - log(p(k) / p(l)) to LLR or not.
        flag_prefactor: A bool. Cost-sensitive or not.
        beta: A float larger than 0. Larger beta leads to 
            more discriminative weights.
            If beta is None, normal, non-cost-sensitive learning
            will be done. If beta = 1, then weights are simply 
            the inverse class frequencies (1 / N_k, 
            where N_k is the sample size of class k).
            If beta = -1, weights = [1,1,...,1] (len=num classes).
            This is useful for non-cost-sensitive learning.
        classwise_sample_sizes: A list of integers. 
            The length is equal to the number of classes.
    Returns:
        gradients: A Tensor or None.
        losses: A list of loss Tensors; namely,
            total_loss: A scalar Tensor or 0 if not calc_grad. 
                The weighted total loss.
            multiplet_loss: A scalar Tensor.
            llr_loss: A scalar Tensor.
            wd_reg: A scalar Tensor.
        logits_concat: A logit Tensor with shape 
            (batch, (duration - order_sprt), order_sprt + 1, num classes). 
            This is the output of 
            datasets.data_processing.sequential_concat(logit_slice, labels_slice).
    Remarks:
        - All the losses below will be calculated if not calc_grad 
          to log them to TensorBoard.
            total_loss 
            multiplet_loss
            llr_loss 
    """
    x_slice, y_slice = sequential_slice(x, y, order_sprt)

    # For training
    if calc_grad:
        with tf.GradientTape() as tape:
            logits_slice = model(x_slice, training)
            logits_concat, y_concat = sequential_concat(
                logits_slice, y_slice, duration)
            total_loss = 0.

            # multiplet_loss, llr_loss
            if param_multiplet_loss != 0:
                multiplet_loss = multiplet_loss_func(logits_slice, y_slice,
                    flag_prefactor, classwise_sample_sizes, beta)
                total_loss += param_multiplet_loss * multiplet_loss
            else:
                multiplet_loss = 0.

            if param_llr_loss != 0:
                lllr = LLLR(logits_concat, y_concat, oblivious, version, flag_mgn,
                        flag_prefactor, flag_prior_ratio, classwise_sample_sizes, beta)
                
                total_loss += param_llr_loss * lllr
            else:
                lllr = 0.

            # wd_reg
            wd_reg = 0.
            if flag_wd:
                for variables in model.trainable_variables:
                    wd_reg += tf.nn.l2_loss(variables)
                    total_loss += param_wd * wd_reg

        gradients = tape.gradient(total_loss, model.trainable_variables)
        losses = [total_loss, multiplet_loss, lllr, wd_reg]

    # For validation and test
    else: 
        logits_slice = model(x_slice, training)
        logits_concat, y_concat = sequential_concat(
            logits_slice, y_slice, duration)
        total_loss = 0.

        # multiplet_loss, llr_loss
        multiplet_loss = multiplet_loss_func(logits_slice, y_slice,
                    flag_prefactor, classwise_sample_sizes, beta)
        lllr = LLLR(logits_concat, y_concat, oblivious, version, flag_mgn,
            flag_prefactor, flag_prior_ratio, classwise_sample_sizes, beta)

        # wd_reg
        wd_reg = 0.
        # for variables in model.trainable_variables:
        #     wd_reg += tf.nn.l2_loss(variables)

        gradients = None
        losses = [0., multiplet_loss, lllr, wd_reg]

    return gradients, losses, logits_concat


def get_gradient_lstm_ver2(model, x, y, training, order_sprt, duration, 
    oblivious, version, flag_wd, flag_mgn, calc_grad, 
    param_multiplet_loss, param_llr_loss, param_wd,
    flag_prefactor=None, classwise_sample_sizes=None, beta=None,
    flag_prior_ratio=None):
    """Calculate loss and/or gradients.
    Args:
        model: A tf.keras.Model object.
        x: A Tensor. A batch of time-series input data 
            without sequential_slice and sequential_concat.
        y: A Tensor. A batch of labels 
            without sequential_slice and sequential_concat.
        training: A boolean. Training flag.
        order_sprt: An int. The order of the SPRT.
        duration: An int. Num of frames in a sequence.
        oblivious: A bool. TANDEMwO or normal TANDEM.
        version: A string, "A", "B", "C", or "D".
        param_multiplet_loss: A float. Loss weight.
        param_llr_loss: A float. Loss weight.
        param_wd: A float. Loss weight.
        flag_wd: A boolean. Weight decay or not.
        flag_mgn: A boolean. Use margin in LLLR or not.
        flag_prior_ratio: A bool. Add the prior ratio term - log(p(k) / p(l)) to LLR or not.
        flag_prefactor: A bool. Cost-sensitive or not.
        beta: A float larger than 0. Larger beta leads to 
            more discriminative weights.
            If beta is None, normal, non-cost-sensitive learning
            will be done. If beta = 1, then weights are simply 
            the inverse class frequencies (1 / N_k, 
            where N_k is the sample size of class k).
            If beta = -1, weights = [1,1,...,1] (len=num classes).
            This is useful for non-cost-sensitive learning.
        classwise_sample_sizes: A list of integers. 
            The length is equal to the number of classes.
    Returns:
        gradients: A Tensor or None.
        losses: A list of loss Tensors; namely,
            total_loss: A scalar Tensor or 0 if not calc_grad. 
                The weighted total loss.
            multiplet_loss: A scalar Tensor.
            llr_loss: A scalar Tensor.
            wd_reg: A scalar Tensor.
        logits_concat: A logit Tensor with shape 
            (batch, (duration - order_sprt), order_sprt + 1, num classes). 
            This is the output of 
            datasets.data_processing.sequential_concat(logit_slice, labels_slice).
    Remarks:
        - All the losses below will be calculated if not calc_grad 
          to log them to TensorBoard.
            total_loss 
            multiplet_loss
            llr_loss 
    """
    x_slice, y_slice = sequential_slice(x, y, order_sprt)

    # For training
    if calc_grad:
        with tf.GradientTape() as tape:
            logits_slice = model(x_slice, training)
            logits_concat, y_concat = sequential_concat(
                logits_slice, y_slice, duration)
            total_loss = 0.

            # multiplet_loss, llr_loss
            if param_multiplet_loss != 0:
                multiplet_loss = multiplet_loss_func(logits_slice, y_slice,
                    flag_prefactor, classwise_sample_sizes, beta)
                total_loss += param_multiplet_loss * multiplet_loss
            else:
                multiplet_loss = 0.

            if param_llr_loss != 0:
                lllr, LLRs = LLLR_ver2(logits_concat, y_concat, oblivious, version, flag_mgn,
                        flag_prefactor, flag_prior_ratio, classwise_sample_sizes, beta)
                
                total_loss += param_llr_loss * lllr
            else:
                lllr = 0.

            # wd_reg
            wd_reg = 0.
            if flag_wd:
                for variables in model.trainable_variables:
                    wd_reg += tf.nn.l2_loss(variables)
                    total_loss += param_wd * wd_reg

        gradients = tape.gradient(total_loss, model.trainable_variables)
        losses = [total_loss, multiplet_loss, lllr, wd_reg]

    # For validation and test
    else: 
        logits_slice = model(x_slice, training)
        logits_concat, y_concat = sequential_concat(
            logits_slice, y_slice, duration)
        total_loss = 0.

        # multiplet_loss, llr_loss
        multiplet_loss = multiplet_loss_func(logits_slice, y_slice,
                    flag_prefactor, classwise_sample_sizes, beta)
        lllr, LLRs = LLLR_ver2(logits_concat, y_concat, oblivious, version, flag_mgn,
            flag_prefactor, flag_prior_ratio, classwise_sample_sizes, beta)

        # wd_reg
        wd_reg = 0.
        # for variables in model.trainable_variables:
        #     wd_reg += tf.nn.l2_loss(variables)

        gradients = None
        losses = [0., multiplet_loss, lllr, wd_reg]

    return gradients, losses, logits_concat, LLRs


def get_loss_fe(model, x, y, flag_wd, training, calc_grad, param_wd):
    """
    Args:
        model: A tf.keras.Model object.
        x: A Tensor with shape=(batch, H, W, C).
        y: A Tensor with shape (batch,).
        flag_wd: A boolean, whether to decay weight here.
        training: A boolean, the training flag.
        calc_grad: A boolean, whether to calculate gradient.
    """
    if calc_grad:
        with tf.GradientTape() as tape:
            logits, bottleneck_feat = model(x, training)
                # (batch, 2) and (batch, final_size)

            xent = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=y, logits=logits))
            
            total_loss = xent

            if flag_wd:
                for variables in model.trainable_variables:
                    total_loss += param_wd * tf.nn.l2_loss(variables)

        gradients = tape.gradient(total_loss, model.trainable_variables)
        losses = [total_loss, xent]

    else:
        logits, bottleneck_feat = model(x, training)
        xent = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y, logits=logits))

        gradients = None 
        losses = [0., xent]

    return gradients, losses, logits, bottleneck_feat

