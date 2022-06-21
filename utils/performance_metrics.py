"""
Remark:
    - logits_concat.shape = (batch, duration - order_sprt, order_sprt + 1, num_classes)
      = (which batch, which sliding window, which component in a sliding window, which class)
"""
import tensorflow as tf
import numpy as np
from utils.misc import restrict_classes

# Functions: logit to confusion matrix
def logits_to_confmx(logits, labels):
    """ Calculate the confusion matrix from logits.
    Args: 
        logits: A logit Tensor with shape (batch, num classes).
        labels: A non-one-hot label Tensor with shape (batch,).
    Returns:
        confmx: A Tensor with shape (num classes, num classes).
    """
    logits_shape = logits.shape # (batch, num classes)
    num_classes = logits_shape[-1]

    # First order_sprt+1 frames
    preds = tf.argmax(logits, axis=-1, output_type=tf.int32) # (batch,)
    confmx = tf.math.confusion_matrix(
        labels=labels, predictions=preds, num_classes=num_classes, dtype=tf.int32)

    return confmx


def multiplet_sequential_confmx(logits_concat, labels_concat):
    """Calculate the confusion matrix for each frame from logits. Lite.
    Args: 
        logits_concat: A logit Tensor with shape 
            (batch, (duration - order_sprt), order_sprt + 1, num_classes). 
            This is the output from
            datasets.data_processing.sequential_concat(logit_slice, y_slice).
        labels_concat: A non-one-hot label Tensor with shape (batch,). 
            This is the output of the function 
            datasets.data_processing.sequential_conclogit_slice, y_slice).
    Return:
        seqconfmx_mult: A Tensor with shape
        (duration, num classes, num classes). This is the series of 
        confusion matrices computed from multiplets.
    Remark:
        e.g., order_sprt = 5, duration = 20:
            confusion matrix for frame001 is given by the 001let of frame001
            confusion matrix for frame002 is given by the 002let of frame002
            ...
            confusion matrix for frame005 is given by the 004let of frame004
            confusion matrix for frame005 is given by the 005let of frame005
            confusion matrix for frame006 is given by the 005let of frame006 computed from frame002-006
            confusion matrix for frame007 is given by the 005let of frame007 computed from frame003-007
            ...
            confusion matrix for frame019 is given by the 005let of frame019 computed from frame015-019
            confusion matrix for frame020 is given by the 005let of frame020 computed from frame016-020
    """
    logits_concat_shape = logits_concat.shape # (batch, (duration - order_sprt), order_sprt + 1, num classes)
    num_classes = logits_concat_shape[-1]

    # First order_sprt+1 frames
    logits_concat_former = logits_concat[:,0,:,:] # (batch, order_sprt + 1, num classes)

    for iter_idx in range(logits_concat_shape[2]):
        preds_former = tf.argmax(logits_concat_former[:, iter_idx, :], 
            axis=-1, output_type=tf.int32) # (batch,)
        if iter_idx == 0:
            seqconfmx_mult = tf.math.confusion_matrix(labels=labels_concat, 
                predictions=preds_former, num_classes=num_classes, dtype=tf.int32)
            seqconfmx_mult = tf.expand_dims(seqconfmx_mult, 0)
        else:
            seqconfmx_mult = tf.concat(
                [seqconfmx_mult, tf.expand_dims(tf.math.confusion_matrix(
                    labels=labels_concat, predictions=preds_former, 
                    num_classes=num_classes, dtype=tf.int32), 0)],
                axis=0
                )

    # Latter duration-order_sprt-1 frames
    logits_concat_latter = logits_concat[:,1:,-1,:] # (batch, duration-order_sprt-1, num classes)

    for iter_idx in range(logits_concat_shape[1]-1):
        preds_latter = tf.argmax(logits_concat_latter[:,iter_idx,:], axis=-1, output_type=tf.int32) # (batch,)
        seqconfmx_mult = tf.concat(
            [seqconfmx_mult, tf.expand_dims(tf.math.confusion_matrix(labels=labels_concat, predictions=preds_latter, num_classes=num_classes, dtype=tf.int32), 0)],
            axis=0
            )

    return seqconfmx_mult


# Functions: confusion matrix to metric
def confmx_to_metrics(confmx): 
    # Superslow for multilclass classification!!!!!!
    # Use seqconfmx_to_metrics instead.
    """Calc confusion-matrix-based performance metrics.
    Args:
        confmx: A confusion matrix Tensor 
            with shape (num classes, num classes).
    Return:
        dict_metrics: A dictionary of dictionaries of performance metrics. 
            E.g., sensitivity of class 3 is dics_metrics["SNS"][3], 
            and accuracy is dict_metrics["ACC"]["original"]
    Remark:
        - SNS: sensitivity, recall, true positive rate
        - SPC: specificity, true negative rate
        - PRC: precision
        - ACC: accuracy
        - BAC: balanced accuracy
        - F1: F1 score
        - GM: geometric mean
        - MCC: Matthews correlation coefficient. May cause overflow.
        - MK: markedness
        - e.g., The classwise accuracy of class i is dict_metric["SNS"][i].
        - "dict_metrics" contains some redundant metrics; 
          e.g., for binary classification, 
          dict_metric["SNS"]["macro"] = dict_metric["BAC"][0] 
          = dict_metric["BAC"][1] = ...
        - Macro-averaged metrics are more robust to class-imbalance 
          than micro-averaged ones, but note that macro-averaged metrics 
          are sometimes equal to be ACC.
        - Most of the micro-averaged metrics are equal to or equivalent to ACC.
    """
    confmx = tf.cast(confmx, tf.int64) # prevent from overflow
    num_classes = confmx.shape[0]
    dict_metrics = {
        "SNS":dict(),
        "SPC":dict(),
        "PRC":dict(),
        "ACC":dict(),
        "BAC":dict(),
        "F1":dict(),
        "GM":dict(),
        "MCC":dict(),
        "MK":dict()
    }
    TP_tot = 0
    TN_tot = 0
    FP_tot = 0
    FN_tot = 0

    # Calc 2x2 confusion matrices out of the multiclass confusion matrix
    for i in range(num_classes):
        # Initialization
        TP = 0
        TN = 0
        FP = 0
        FN = 0

        # Calc TP, TN, FP, FN for class i
        TP = confmx[i,i]
        for j in range(num_classes):
            if j == i:
                continue
            FP += confmx[j,i]
            FN += confmx[i,j]
            for k in range(num_classes):
                if k == i:
                    continue
                TN += confmx[j,k]

        # Calc performance metrics of class i
        dict_metrics["SNS"][i] = TP/(TP+FN) if TP+FN != 0 else 0.
        dict_metrics["SPC"][i] = TN/(TN+FP) if TN+FP != 0 else 0.
        dict_metrics["PRC"][i] = TP/(TP+FP) if TP+FP != 0 else 0.
        dict_metrics["ACC"][i] = (TP+TN)/(TP+FN+TN+FP) if TP+FN+TN+FP != 0 else 0.
        dict_metrics["BAC"][i] = (dict_metrics["SNS"][i] + dict_metrics["SPC"][i])/2
        dict_metrics["F1"][i] = \
            2 * (dict_metrics["PRC"][i] * dict_metrics["SNS"][i]) \
            / (dict_metrics["PRC"][i] + dict_metrics["SNS"][i]) \
            if dict_metrics["PRC"][i] + dict_metrics["SNS"][i] != 0 else 0.
        dict_metrics["GM"][i] = np.sqrt(dict_metrics["SNS"][i] * dict_metrics["SPC"][i])
        dict_metrics["MCC"][i] = \
            ((TP*TN) - (FP*FN)) / (np.sqrt((TP + FP ) * (TP + FN) * (TN + FP) * (TN + FN)))\
            if (TP+FP)*(TP+FN)*(TN+FP)*(TN+FN) != 0 else 0.
        dict_metrics["MK"][i] = \
            dict_metrics["PRC"][i] + (TN / (TN + FN)) - 1 \
                if TN + FN != 0 else dict_metrics["PRC"][i] - 1

        TP_tot += TP
        TN_tot += TN
        FP_tot += FP
        FN_tot += FN

    # Calc micro- and macro- averaged metrics
    # sometimes returns nan. please fix it
    dict_metrics["SNS"]["macro"] = np.mean([dict_metrics["SNS"][i] for i in range(num_classes)])
    dict_metrics["SNS"]["micro"] = TP_tot/(TP_tot+FN_tot) if TP_tot+FN_tot != 0 else 0. # = original ACC. 
    dict_metrics["SPC"]["macro"] = np.mean([dict_metrics["SPC"][i] for i in range(num_classes)])
    dict_metrics["SPC"]["micro"] = TN_tot/(TN_tot+FP_tot) if TN_tot+FP_tot != 0 else 0.
    dict_metrics["PRC"]["macro"] = np.mean([dict_metrics["PRC"][i] for i in range(num_classes)])
    dict_metrics["PRC"]["micro"] = TP_tot/(TP_tot+FP_tot) if TP_tot+FP_tot != 0 else 0. # = original ACC. 
    dict_metrics["ACC"]["macro"] = np.mean([dict_metrics["ACC"][i] for i in range(num_classes)])
    dict_metrics["ACC"]["micro"] = (TP_tot+TN_tot)/(TP_tot+FN_tot+TN_tot+FP_tot) if TP_tot+FN_tot+TN_tot+FP_tot != 0 else 0.
    dict_metrics["ACC"]["original"] = ((num_classes/2) * dict_metrics["ACC"]["micro"]) - ((num_classes-2)/2)
    dict_metrics["BAC"]["macro"] = np.mean([dict_metrics["BAC"][i] for i in range(num_classes)])
    dict_metrics["BAC"]["micro"] = (dict_metrics["SNS"]["micro"] + dict_metrics["SPC"]["micro"])/2
    dict_metrics["F1"]["macro"] = np.mean([dict_metrics["F1"][i] for i in range(num_classes)])
    dict_metrics["F1"]["micro"] = 2*(dict_metrics["PRC"]["micro"] * dict_metrics["SNS"]["micro"]) / (dict_metrics["PRC"]["micro"] + dict_metrics["SNS"]["micro"]) if dict_metrics["PRC"]["micro"] + dict_metrics["SNS"]["micro"] != 0 else 0.# = original ACC. 
    dict_metrics["GM"]["macro"] = np.mean([dict_metrics["GM"][i] for i in range(num_classes)])
    dict_metrics["GM"]["micro"] = np.sqrt(dict_metrics["SNS"]["micro"] * dict_metrics["SPC"]["micro"])
    dict_metrics["MCC"]["macro"] = np.mean([dict_metrics["MCC"][i] for i in range(num_classes)])
    dict_metrics["MCC"]["micro"] = ((TP_tot*TN_tot) - (FP_tot*FN_tot))/(np.sqrt( (TP_tot+FP_tot)*(TP_tot+FN_tot)*(TN_tot+FP_tot)*(TN_tot+FN_tot) )) if (TP_tot+FP_tot)*(TP_tot+FN_tot)*(TN_tot+FP_tot)*(TN_tot+FN_tot) != 0 else 0.
    dict_metrics["MK"]["macro"] = np.mean([dict_metrics["MK"][i] for i in range(num_classes)])
    dict_metrics["MK"]["micro"] = dict_metrics["PRC"]["micro"] + (TN_tot/(TN_tot+FN_tot)) - 1 if TN_tot+FN_tot != 0 else 0. 

    return dict_metrics


def seqconfmx_to_metrics(seqconfmx): # used in plot_SATC_casula.ipynb as of July 7, 2020
    """Calc confusion-matrix-based performance metrics.
    Important Note (February 24th, 2021): 
        If no example belonging to a class comes in `seqconfmx`, 
        i.e., all the entries in the raw corresponding to that class are zero,
        then the classwise metric of that class is assumed to be ZERO. 
        However, the macro-averaged metric IGNORES such empty classes (V2), 
        while in `seqconfmx_to_metrics` (V1), the macro-averaged metrics 
        assume that the classwise metrics of the empty classes are ZERO 
        (i.e., do not ignore them), which may significantly degradate 
        the macro-averaged metrics (e.g., when the sample size used for `seqconfmx` 
        is much smaller than the number of classes).
    Args:
        seqconfmx: A series of confusion matrix Tensors 
            with shape (series length (batch), num classes, num classes).
    Return:
        dict_metrics: A dictionary of performance metrics including
            classwise, macro-averaged, and micro-averaged metrics. 
    Remarks:
        - Examples:
            dics_metrics["SNS"][k] = sensitivity of class 3, where
                k = 0, 1, 2, ..., num classes - 1.
            dict_metrics["SNS"][num classes] = macro-averaged sensitivity.
            dict_metrics["SNS"][num classes + 1] = micro-averaged sensitivity, which
                is equal to accuracy.

        - Currently, only "SNS"s are calculated. Thus dics_metrics.keys() = ["SNS",].
          But adding another metric is easy.
    """
    seqconfmx = tf.cast(seqconfmx, tf.float64) # avoids overflow
    dict_metrics = dict()

    # Calc 2x2 confusion matrices out of the multiclass confusion matrix
    TP = tf.linalg.diag_part(seqconfmx)
        # (batch, num cls)
    FP = tf.reduce_sum(seqconfmx, axis=1) - TP
        # (batch, num cls)
    FN = tf.reduce_sum(seqconfmx, axis=2) - TP
        # (batch, num cls)
    TN = tf.reduce_sum(seqconfmx, axis=1)
        # (batch, num cls)
    TN = tf.reduce_sum(TN, axis=1, keepdims=True)
        # (batch, 1)
    TN -= TP + FP + FN
        # (batch, num cls)
    TP_tot = tf.reduce_sum(TP, axis=1, keepdims=True)
    TN_tot = tf.reduce_sum(TN, axis=1, keepdims=True)
    FP_tot = tf.reduce_sum(FP, axis=1, keepdims=True)
    FN_tot = tf.reduce_sum(FN, axis=1, keepdims=True)
        # (batch, 1)
    
    # Calc classwise, macro-ave, micro-ave metrics
    SNS = TP / (TP + FN + 1e-12)
        # (batch, num cls)
    macroave_SNS = tf.reduce_mean(SNS, axis=1, keepdims=True)
        # (batch, 1)
    microave_SNS = TP_tot / (TP_tot + FN_tot + 1e-12)
        # (batch, 1)

    # Concat
    dict_metrics["SNS"] = tf.concat(
        [SNS, macroave_SNS, microave_SNS], axis=1)
        # (batch, num cls + 2)

    return dict_metrics


def seqconfmx_to_metricsV2(seqconfmx): # Used in training codes created after that of UCF101. 
    """ Calc confusion-matrix-based performance metrics.
        V2 supports accuracy and implements a different calc criterion 
        of the macro-averaged metrics. V2's output is np.ndarray, not Tensor.
    Important Note (February 24th, 2021): 
        If no example belonging to a class comes in `seqconfmx`, 
        i.e., all the entries in the raw corresponding to that class are zero,
        then the classwise metric of that class is assumed to be ZERO. 
        However, the macro-averaged metric IGNORES such empty classes (V2), 
        while in `seqconfmx_to_metrics` (V1), the macro-averaged metrics 
        assume that the classwise metrics of the empty classes are ZERO 
        (i.e., do not ignore them), which may significantly degradate 
        the macro-averaged metrics (e.g., when the sample size used for `seqconfmx` 
        is much smaller than the number of classes).
    Args:
        seqconfmx: A series of confusion matrix Tensors 
            with shape (series length (arbitrary), num classes, num classes).
    Return:
        dict_metrics: A dictionary of performance metrics including
            classwise, macro-averaged, and micro-averaged metrics. 
    Remarks:
        - Examples:
            dics_metrics["SNS"][k] = sensitivity of class 3, where
                k = 0, 1, 2, ..., num classes - 1.
            dict_metrics["SNS"][num classes] = macro-averaged sensitivity.
            dict_metrics["SNS"][num classes + 1] = micro-averaged sensitivity, which
                is equal to accuracy.
    """
    duration = seqconfmx.shape[0]
    seqconfmx = tf.cast(seqconfmx, tf.float64) # avoids overflow
    dict_metrics = dict()
    classwise_sample_size = tf.reduce_sum(seqconfmx, axis=2)
        # shape = (duration, num cls)
    mask = tf.where(tf.not_equal(classwise_sample_size, 0))
        # A Tensor of integer indices.
        # shape = (num of classes with non-zero sample sizes, 2)
        # 2 means [raw index, column index]
        # E.g.,  
        #    <tf.Tensor: id=212, shape=(1250, 2), dtype=int64, numpy=
        #    array([[  0,   1],
        #        [  0,   5],
        #        [  0,   6],
        #        ...,
        #        [ 49,  90],
        #        [ 49,  99],
        #        [ 49, 100]])>
        # Usage:
        #    NEW_TENSOR = tf.reshape(tf.gather_nd(TENSOR, mask), [duration, -1]),
        #    where TENSOR.shape is (duration, num_classes). 
        #    NEW_TENSOR has shape (duration, num of classes with non-zero sample sizes).
        #    TENSOR can be SNS, TP, TN, FP, FN, etc., as shown below.

    # Calc 2x2 confusion matrices out of the multiclass confusion matrix
    TP = tf.linalg.diag_part(seqconfmx)
        # (duration, num cls)
    FP = tf.reduce_sum(seqconfmx, axis=1) - TP
        # (duration, num cls)
    FN = tf.reduce_sum(seqconfmx, axis=2) - TP
        # (duration, num cls)
    TN = tf.reduce_sum(seqconfmx, axis=1)
        # (duration, num cls)
    TN = tf.reduce_sum(TN, axis=1, keepdims=True)
        # (duration, 1)
    TN -= TP + FP + FN
        # (duration, num cls)
    TP_tot = tf.reduce_sum(
        tf.reshape(tf.gather_nd(TP, mask), [duration, -1]), 
        axis=1, 
        keepdims=True)
        # (duration, 1)
    TN_tot = tf.reduce_sum(
        tf.reshape(tf.gather_nd(TN, mask), [duration, -1]), 
        axis=1, 
        keepdims=True)
        # (duration, 1)
    FP_tot = tf.reduce_sum(
        tf.reshape(tf.gather_nd(FP, mask), [duration, -1]), 
        axis=1, 
        keepdims=True)
        # (duration, 1)
    FN_tot = tf.reduce_sum(
        tf.reshape(tf.gather_nd(FN, mask), [duration, -1]), 
        axis=1, 
        keepdims=True)
        # (duration, 1)
    
    # Sensitivity (Recall, Classwise accuracy)
    ############################################################
    # Calc classwise, macro-ave, micro-ave metrics
    SNS = TP / (TP + FN + 1e-12)
        # (duration, num cls)
    macroave_SNS = tf.reduce_mean(
        tf.reshape(tf.gather_nd(SNS, mask), [duration, -1]), 
        axis=1, 
        keepdims=True)
        # (duration, 1)
    microave_SNS = TP_tot / (TP_tot + FN_tot + 1e-12)
        # (duration, 1)

    # Concat
    dict_metrics["SNS"] = np.concatenate(
        [SNS.numpy(), macroave_SNS.numpy(), microave_SNS.numpy()], axis=1)
        # (duration, num cls + 2)

    # Hamming accuracy
    ############################################################
    # ACC = (TP + TN) / (TP + TN + FP + FN + 1e-12)
    # macroave_ACC = tf.reduce_mean(
    #     tf.reshape(tf.gather_nd(ACC, mask), [duration, -1]), 
    #     axis=1, 
    #     keepdims=True)
    #     # (duration, 1)
    # microave_ACC = (TP_tot + TN_tot) / (TP_tot + TN_tot + FP_tot + FN_tot + 1e-12)
    # #assert tf.reduce_all(tf.equal(tf.cast(macroave_ACC, tf.float32), tf.cast(microave_ACC, tf.float32)))
    
    # dict_metrics["ACC"] = np.concatenate(
    #     [ACC.numpy(), macroave_ACC.numpy(), microave_ACC.numpy()], axis=1)
    #     # (duration, num cls + 2)

    # Jaccard index
    ############################################################
    JAC = (TP) / (TP + FP + FN + 1e-12)
    macroave_JAC = tf.reduce_mean(
        tf.reshape(tf.gather_nd(JAC, mask), [duration, -1]), 
        axis=1, 
        keepdims=True)
        # (duration, 1)
    microave_JAC = (TP_tot) / (TP_tot + FP_tot + FN_tot + 1e-12)
    
    dict_metrics["JAC"] = np.concatenate(
        [JAC.numpy(), macroave_JAC.numpy(), microave_JAC.numpy()], axis=1)
        # (duration, num cls + 2)
    dict_metrics["exJAC"] = TP_tot / (TP_tot + FP_tot + 1e-12) 
        # = 1/N * sum_i 1(pred_i = label_i), i.e., ordinary "accuracy". 
        # shape = (duration, 1)

    return dict_metrics


def confmx_to_macrec(confmx): 
    """ confmx -> macroaveraged recall (averaged per-class accuracy).
    Args:
        confmx: A Tensor with shape (num classes, num classes)
    Returns:
        bac: A scalar Tensor, balanced accuracy.
            Independend of num of classes (>= 2).
    """
    dict_metrics = seqconfmx_to_metrics(tf.expand_dims(confmx, axis=0))
    rec = dict_metrics["SNS"][0]
    macrec = rec[-2]
    return macrec


################################################
# Functions For SPRT
################################################

# Functions: Calc LLRs (logit -> LLR)
def cost_weight_generator(classwise_sample_sizes, beta):
    """
    Instructions:
    - TL;DR: beta = 1 - 1e-2, 1 - 1e-3, 1 - 1e-4, and 1 - 1e-5 are
      recommended for UCF101-50-H-W and UCF101-150-H-W, while
      beta = 1 - 1e-2 and 1 - 1e-3 are recommended for HMDB51-79-H-W.
    - More generally, beta = 1 - 1e-2, 1 - 1e-3, 1 - 1e-4, and 1 - 1e-5 are
      recommended for classwise sample sizes < 10k,
      while beta = 1 - 1e-6 and beta = 1e-7 are also recommended for
      classwise sample sizes > 100k.
    - Larger beta leads to more discriminative weights.
    - beta = 1 - 1e-1, 1 - 1e-2, 1 - 1e-3, ..., up to 1 - 1e-7 are allowed; 
      beta = 1 - 1e-8 = 0.99999999 is rounded to 1 in float32 calculations.
    - When beta = 1 - 1e-1, the weights are 0.15, 0.1, 0.1, and 0.1 
      for classwise sample size 10, 100, 10k, and 100k, respectively;
      thus beta = 1 - 1e-1 is almost meaningless.
    - beta = 1 - 1e-5, 1 - 1e-6, and 1 - 1e-7 give almost the same weights 
      for classwise sample sizes less than 10k.
    =============================================================
    Args:
        classwise_sample_sizes: A list of integers. 
            The length is equal to the number of classes.
        beta: A float larger than 0. Larger beta leads to more discriminative weights.
            If beta = 1, weights are simply the inverse class frequencies (1 / N_k,
            where N_k is the sample size of class k).
            If beta = -1, weights = [1,1,...,1] (len=num classes).
            This is useful for non-cost-sensitive learning.
    Returns:
        weights: A Tensor of float32 numbers with shape (num classes,).
    Remarks:
        A difference from the original paper is
        weights /= tf.reduce_sum(weights) + 1e-12.
        This ensures that sum_k weight[k] = 1, i.e., weights are normalized,
        which is not guaranteed in the origianl paper. This normalization can
        facilitate balancing the learning rate and `beta`.
    """
    assert beta < 1 - 1e-8 or beta == 1, \
        "beta should be less than 0.9999999 or 1; otherwise we face the loss of trailing digits."
    beta = np.float32(beta)
    
    if beta == -1:
        weights = tf.constant([1] * len(classwise_sample_sizes), dtype=tf.float32)
            # shape = (num_classes, )

    elif beta == 1:
        weights = 1 / (tf.constant(classwise_sample_sizes, dtype=tf.float32) + 1e-12)
            # shape = (num_classes,)
        weights /= tf.reduce_sum(weights) + 1e-12
            # Normalization. Not present in the original paper
            
    else:
        weights = (1. - beta) / (1. - tf.pow(beta, classwise_sample_sizes) + 1e-12) 
            # shape = (num_classes,)
        weights /= tf.reduce_sum(weights) + 1e-12
            # Normalization. Not present in the original paper
    
    return weights


def calc_llrs(logits_concat, flag_prior_ratio=None, classwise_sample_sizes=None, beta=None): # used in LLLR
    """ Calculate the frame-by-frame log-likelihood ratio matrices.
        Used to calculate LLR(x^(1), ..., x^(t)) 
        with N-th order posteriors (the N-th order TANDEM formula).
    Args:
        logits_concat: A logit Tensor with shape 
            (batch, (duration - order_sprt), order_sprt + 1, num classes). 
            This is the output from 
            datasets.data_processing.sequential_concat(
            logit_slice, labels_slice).
        flag_prior_ratio: A bool. Add the prior ratio term - log(p(k) / p(l)) or not.
        classwise_sample_sizes: A list of intergers. Classwise sample sizes.
        beta: A float larger than 0. Larger beta leads to more discriminative weights.
            If beta = 1, weights are simply the inverse class frequencies (1 / N_k,
            where N_k is the sample size of class k).
            If beta = -1, weights = [1,1,...,1] (len=num classes).
            This is useful for non-cost-sensitive learning.
    Returns:
        llr_matrix: A Tensor 
            with shape (batch, duration, num classes, num classes).
            matrix[i, j] = log(p_i/p_j), where p_i and p_j are the likelihood
            of class i and that of j, resp.
            They are anti-symmetric matrices.
    Remarks:
        - The LLRs returned are the LLRs used in the 
          "order_sprt"-th order SPRT; the LLRs unnecessary to calculate the 
          "order_sprt"-th order SPRT are not included. 
        - "duration" and "order_sprt" are automatically calculated 
          using logits_concat.shape.
    An examle of assert checks performed for operation confirmation:
        # Given random logits_concat, including logits_concat with all entries = 0,
        llrs1 = calc_llrs(logits_concat, flag_prior_ratio=False).numpy()
        llrs2 = calc_llrs(logits_concat, flag_prior_ratio=True, classwise_sample_sizes=classwise_sample_sizes).numpy()
        llrs3 = calc_oblivious_llrs(logits_concat, flag_prior_ratio=False).numpy()
        llrs4 = calc_oblivious_llrs(logits_concat, flag_prior_ratio=True, classwise_sample_sizes=classwise_sample_sizes).numpy()
        for b in range(batch_size):
            for t in range(duration):
                for i in range(num_classes):
                    for j in range(num_classes):
                        if i == j:
                            assert llrs1[b, t, i, j] == llrs2[b, t, i, j] and llrs2[b, t, i, j] == 0
                            assert llrs3[b, t, i, j] == llrs4[b, t, i, j] and llrs4[b, t, i, j] == 0
                        else:
                            if classwise_sample_sizes[i] >=classwise_sample_sizes[j]:
                                assert llrs1[b, t, i, j] >= llrs2[b, t, i, j] 
                                assert llrs3[b, t, i, j] >= llrs4[b, t, i, j]
                            else:
                                assert llrs1[b, t, i, j] < llrs2[b, t, i, j] 
                                assert llrs3[b, t, i, j] < llrs4[b, t, i, j]                    
    """
    logits_concat_shape = logits_concat.shape
    order_sprt = int(logits_concat_shape[2] - 1)
    duration = int(logits_concat_shape[1] + order_sprt)
    num_classes = int(logits_concat_shape[3])
    assert num_classes > 1, "num_classes={} must > 1".format(num_classes)
    logits1 = tf.expand_dims(logits_concat, axis=-1)
        # (batch, duration - order, order + 1, num cls, 1)
    logits2 = tf.expand_dims(logits_concat, axis=-2)
        # (batch, duration - order, order + 1, 1, num cls)
    list_llrs = []

    # i.i.d. LLR (for 0th order SPRT)
    if order_sprt == 0:
        llrs_all_frames = logits1[:, :, order_sprt, :, 0:]\
            - logits2[:, :, order_sprt, 0:, :] 
            # (batch, duration, num cls, num cls) 
        for iter_frame in range(duration):
            llrs = tf.reduce_sum(
                llrs_all_frames[:, :iter_frame+1, :, :], 1) 
                # (batch, num cls, num cls)
            list_llrs.append(tf.expand_dims(llrs, 1))

    # N-th order LLR (for N-th order SPRT)
    else:
        for iter_frame in range(duration):
            if iter_frame < order_sprt + 1:
                llrs = logits1[:, 0, iter_frame, :, 0:] - logits2[:, 0, iter_frame, 0:, :] 
                    # (batch, num cls, num cls)
                list_llrs.append(tf.expand_dims(llrs, 1))

            else:
                llrs1 = logits1[:, :iter_frame - order_sprt + 1, order_sprt, :, 0:]\
                    - logits2[:, :iter_frame - order_sprt + 1, order_sprt, 0:, :] 
                    # (batch, iter_frame-order_sprt, num cls, num cls)
                llrs1 = tf.reduce_sum(llrs1, 1) # (batch, num cls, num cls)
                llrs2 = logits1[:, 1:iter_frame - order_sprt + 1, order_sprt-1, :, 0:]\
                    - logits2[:, 1:iter_frame - order_sprt + 1, order_sprt-1, 0:, :] 
                    # (batch, iter_frame-order_sprt-1, num cls, num cls)
                llrs2 = tf.reduce_sum(llrs2, 1) # (batch, num cls, num cls)
                llrs = llrs1 - llrs2 # (batch, num cls, num cls)
                list_llrs.append(tf.expand_dims(llrs, 1))

    llr_matrix = tf.concat(list_llrs, 1) # (batch, duration, num cls, num cls)
    
    # Add prior ratio term
    if flag_prior_ratio: 
        assert classwise_sample_sizes is not None
        weights = cost_weight_generator(classwise_sample_sizes, beta)
        prior = tf.reshape(weights, [1, 1, num_classes])
            # (1, 1, num cls, )
        llr_matrix += - tf.math.log(tf.expand_dims(prior, axis=2) / tf.expand_dims(prior, axis=3))
            # (batch, duration, num cls, num cls)

    return llr_matrix


def calc_oblivious_llrs(logits_concat, flag_prior_ratio=None, classwise_sample_sizes=None, beta=None): # used in LLLR
    """ Calculate the frame-by-frame log-likelihood ratio matrices.
        Used to calculate LLR(x^(t-N), ..., x^(t)) 
        i.e., (the N-th order TANDEMsO formula).
    Args:
        logits_concat: A logit Tensor with shape 
            (batch, (duration - order_sprt), order_sprt + 1, num classes). 
            This is the output from 
            datasets.data_processing.sequential_concat(
            logit_slice, labels_slice).
        beta: A float larger than 0. Larger beta leads to more discriminative weights.
            If beta = 1, weights are simply the inverse class frequencies (1 / N_k,
            where N_k is the sample size of class k).
            If beta = -1, weights = [1,1,...,1] (len=num classes).
            This is useful for non-cost-sensitive learning.
    Returns:
        llr_matrix: A Tensor 
            with shape (batch, duration, num classes, num classes).
            matrix[i, j] = log(p_i/p_j), where p_i and p_j are the likelihood
            of class i and that of j, resp.
            They are anti-symmetric matrices.
    Remarks:
        - The LLRs returned are the LLRs used in the 
          "order_sprt"-th order SPRT; the LLRs unnecessary to calculate the 
          "order_sprt"-th order SPRT are not included. 
        - "duration" and "order_sprt" are automatically calculated 
          from logits_concat.shape.
    An examle of assert checks for operation confirmation:
        # Given random logits_concat and logits_concat with all entries = 0,
        llrs1 = calc_llrs(logits_concat, flag_prior_ratio=False).numpy()
        llrs2 = calc_llrs(logits_concat, flag_prior_ratio=True, classwise_sample_sizes=classwise_sample_sizes, beta=1).numpy()
        llrs3 = calc_oblivious_llrs(logits_concat, flag_prior_ratio=False).numpy()
        llrs4 = calc_oblivious_llrs(logits_concat, flag_prior_ratio=True, classwise_sample_sizes=classwise_sample_sizes, beta=1).numpy()
        for b in range(batch_size):
            for t in range(duration):
                for i in range(num_classes):
                    for j in range(num_classes):
                        if i == j:
                            assert llrs1[b, t, i, j] == llrs2[b, t, i, j] and llrs2[b, t, i, j] == 0
                            assert llrs3[b, t, i, j] == llrs4[b, t, i, j] and llrs4[b, t, i, j] == 0
                        else:
                            if classwise_sample_sizes[i] >=classwise_sample_sizes[j]:
                                assert llrs1[b, t, i, j] >= llrs2[b, t, i, j] 
                                assert llrs3[b, t, i, j] >= llrs4[b, t, i, j]
                            else:
                                assert llrs1[b, t, i, j] < llrs2[b, t, i, j] 
                                assert llrs3[b, t, i, j] < llrs4[b, t, i, j]     
    """
    logits_concat_shape = logits_concat.shape
    order_sprt = int(logits_concat_shape[2] - 1)
    duration = int(logits_concat_shape[1] + order_sprt)
    num_classes = int(logits_concat_shape[3])
    assert num_classes > 1, "num_classes={} must > 1".format(num_classes)

    logits1 = tf.expand_dims(logits_concat, axis=-1)
        # (batch, duration - order, order + 1, num cls, 1)
    logits2 = tf.expand_dims(logits_concat, axis=-2)
        # (batch, duration - order, order + 1, 1, num cls)
    list_llrs = []

    # i.i.d. SPRT (0th-order SPRT)
    if order_sprt == 0:
        llrs_all_frames = logits1[:, :, order_sprt, :, 0:]\
            - logits2[:, :, order_sprt, 0:, :] 
            # (batch, duration, num cls, num cls) 
        llr_matrix = llrs_all_frames # oblivious!!

    # N-th order LLR (for N-th order oblivious SPRT)
    else:
        for iter_frame in range(duration):
            if iter_frame < order_sprt + 1:
                llrs = logits1[:, 0, iter_frame, :, 0:] - logits2[:, 0, iter_frame, 0:, :] 
                    # (batch, num cls, num cls)
                list_llrs.append(tf.expand_dims(llrs, 1))

            else:
                llrs1 = logits1[:, iter_frame - order_sprt, order_sprt, :, 0:]\
                    - logits2[:, iter_frame - order_sprt, order_sprt, 0:, :] 
                    # (batch, num cls, num cls)
                    # removed two colons and two "+1" to be oblivious
                #llrs1 = tf.reduce_sum(llrs1, 1) # (batch, num cls, num cls)
                #llrs2 = logits1[:, 1:iter_frame - order_sprt + 1, order_sprt-1, :, 0:]\
                #    - logits2[:, 1:iter_frame - order_sprt + 1, order_sprt-1, 0:, :] 
                #    # (batch, iter_frame-order_sprt-1, num cls, num cls)
                #llrs2 = tf.reduce_sum(llrs2, 1) # (batch, num cls, num cls)
                llrs = llrs1 #- llrs2 # (batch, num cls, num cls)
                list_llrs.append(tf.expand_dims(llrs, 1))
    
        llr_matrix = tf.concat(list_llrs, 1) # (batch, duration, num cls, num cls)

    # Add prior ratio term
    if flag_prior_ratio: 
        assert classwise_sample_sizes is not None
        weights = cost_weight_generator(classwise_sample_sizes, beta)
        prior = tf.reshape(weights, [1, 1, num_classes])
            # (1, 1, num cls, )
        llr_matrix += - tf.math.log(tf.expand_dims(prior, axis=2) / tf.expand_dims(prior, axis=3))
            # (batch, duration, num cls, num cls)
        
    return llr_matrix


# Functions: threshold (LLR -> thresh)
def threshold_generator(llrs, num_thresh, sparsity): # used in LLLR
    """ Generates sequences of sigle-valued threshold matrices.        
    Args:
        llrs: A Tensor with shape 
            [batch, duration, num classes, num classes].
            Anti-symmetric matrix.
        num_thresh: An integer, the number of threholds.
            1 => thresh = minLLR
            2 => thresh = minLLR, maxLLR
            3 => thresh = minLLR, (minLLR+maxLLR)/2, maxLLR
            ... (linspace float numbers).
        sparsity: "linspace", "logspace", "unirandom", or "lograndom". 
            Linearly spaced, logarithmically spaced, uniformly random,
            or log-uniformly random thresholds are generated
            between min LLR and max LLR.    
    Returns:
        thresh: A Tensor with shape 
            (num_thresh, duration, num classes, num classes).
            In each matrix, 
            diag = 0, and off-diag shares a single value > 0.
            Sorted in ascending order of the values.
    Remarks:
        - The threshold values are in [min |LLR| (!= 0), max |LLR|].
        - For reference, we show the Wald's approximation:
          If alpha is a float in (0, 0.5) (FPR) and 
          beta is a float in (0, 0.5) (FNR),
          then thresh 
          = [np.log(beta/(1-alpha)), np.log((1-beta)/alpha)].
    """
    llrs_shape = llrs.shape
    num_classes = llrs_shape[-1]
    duration = llrs_shape[1]

    # Remove 0 LLRs in off-diags
    tri = tf.ones_like(llrs) # memory consuming
    triu = tf.linalg.band_part(tri, 0, -1) # Upper triangular part.
    tril = tf.linalg.band_part(tri, -1, 0) # Lower triangular part.
    llrs -= 1e-12 * (triu - tril) 

    # Calc non-zero max and min of |LLRs|
    llrs_abs = tf.abs(llrs)
    llrs_max = tf.reduce_max(llrs_abs)
        # max |LLRs|
    tmp = tf.linalg.tensor_diag([1.] * num_classes) * llrs_max
    tmp = tf.reshape(tmp, [1, 1, num_classes, num_classes])
    llrs_min = tf.reduce_min(llrs_abs + tmp)
        # strictly positive (non-zero) minimum of LLRs

    # Single-valued threshold matrix
    if sparsity == "linspace":
        thresh = tf.linspace(llrs_min, llrs_max, num_thresh)    
            # (num thresh,)

    elif sparsity == "logspace":
        thresh = tf.math.exp(
            tf.linspace(
                tf.math.log(llrs_min), 
                tf.math.log(llrs_max), 
                num_thresh))
            # (num thresh,)

    elif sparsity == "unirandom":
        thresh = tf.random.uniform(shape=[num_thresh])
        thresh = tf.sort(thresh)
        thresh = ((llrs_max - llrs_min) * thresh) + llrs_min
            # (num thresh,), ascending order

    elif sparsity == "lograndom":
        thresh = tf.random.uniform(shape=[num_thresh])
        thresh = tf.sort(thresh)
        thresh = tf.math.exp(
            ((tf.math.log(llrs_max) - tf.math.log(llrs_min)) * thresh)\
                + tf.math.log(llrs_min))
            # (num thresh,). Ascending order.

    else:
        raise ValueError 

    thresh = tf.reshape(thresh, [num_thresh, 1, 1, 1])
    thresh = tf.tile(thresh, [1, duration, num_classes, num_classes])
        # (num thresh, duration, num cls, num cls)
    mask = tf.linalg.tensor_diag([-1.] * num_classes) + 1
    thresh *= mask
        # Now diag = 0.
    thresh += mask * 1e-11
        # Avoids 0 threholds, which may occur
        # when logits for different classes have the same value,
        # e.g., 0, due to loss of significance.
        # This operation may cause sparsity of SAT curve 
        # if llr_min is << 1e-11, but such a case is ignorable 
        # in practice, according to my own experience. 

    return thresh


def thresh_sanity_check(thresh_mtx):
    """ Sanity check of the threshold matrix.
    Args:
        thresh_mtx: A Tensor with shape 
            (num thresholds, duration, num class, num class).
    """
    """ check this works for batch, duration, num thresh = 1"""
    num_classes = thresh_mtx.shape[2]
    
    for i in range(num_classes):
        if not tf.reduce_all(thresh_mtx[:, :, i, i] == 0):
            raise ValueError(
                "The diag elements of thresh_mtx must be 0." +
                "If Nan or inf is in thresh, maybe min|LLR| = 0 " + 
                "or max|LLR|=inf (or >> 200)." +
                "\nNow thresh_mtx = {}".format(thresh_mtx)
                )

    tmp = tf.linalg.tensor_diag([1.] * num_classes)
    tmp = tf.reshape(tmp, [1, 1, num_classes, num_classes])
    tmp_th = thresh_mtx + tmp
    if not tf.reduce_all(tmp_th > 0):
        raise ValueError(
            "The off-diag elements of thresh_mtx must be positive."+
            "\nNow thresh_mtx = {}".format(thresh_mtx)
            )


# Function: Matrix SPRT (LLR, thresh -> confmx)
def truncated_MSPRT(llr_mtx, labels_concat, thresh_mtx):
    """ Truncated Matrix-SPRT.
    Args:
        llr_mtx: A Tensor with shape 
            (batch, duration, num classes, num classes).
            Anti-symmetric matrices.
        labels_concat: A Tensor with shape (batch,).
        thresh_mtx: A Tensor with shape 
            (num thresholds, duration, num class, num class).
            Diag must be 0. Off diag must be strictly positive.
            To be checked in this function.
    Returns:
        confmx: A Tensor with shape (num thresh, classes, num classes).
            Confusion matrix.
        mht: A Tensor with shape (num thresh,). 
            Mean hitting time.
        vht: A Tensor with shape (num thresh,).
            Variance of hitting times.
        trt: A Tensor with shape (num thresh,). 
            Truncation rate.
    """
    """ check shape match, then values """
    """ care about exactly zero LLRs: Done """
    thresh_mtx_shape = thresh_mtx.shape
    num_thresh = thresh_mtx_shape[0]
    duration = thresh_mtx_shape[1]
    num_classes = thresh_mtx_shape[2]
    batch_size = llr_mtx.shape[0]

    # Sanity check of thresholds
    thresh_sanity_check(thresh_mtx)
    thresh_mtx = tf.cast(thresh_mtx, dtype=tf.float32)

    # Reshape and calc scores
    llr_mtx = tf.expand_dims(llr_mtx, 0) 
        # (1, batch, duration, num cls, num cls)
        # to admit the num-thresh axis.
    thresh_mtx = tf.expand_dims(thresh_mtx, 1)
        # (num thresh, 1, duration, num cls, num cls)
        # to admit the batch axis.
    tri = tf.ones_like(llr_mtx)
    triu = tf.linalg.band_part(tri, 0, -1) # Upper triangular part.
    tril = tf.linalg.band_part(tri, -1, 0) # Lower triangular part.
    llr_mtx -= 1e-12 * (triu - tril) 
        # (1, batch, duration , num cls, num cls)
        # To avoid double hit due to the values exactly equal to zero 
        # in scores or when doing truncation, LLRs of the last frame.
    scores = tf.reduce_min(llr_mtx - thresh_mtx, -1)
        # (num thresh, batch, duration, num cls)
        # Values are non-positive.
    """ assert 1: for each thresh, batch, and duration, 
                  the num of 0 is 0 or at most 1 in the last axis direction
        assert 2: values <= 0
    """

    # Calc all predictions and waits
    preds_all = tf.sign(scores) + 1
        # 0:wait, 1:hit (one-hot vector)
        # (num thresh, batch, duration, num cls)
    """assert actually one-hot"""

    # Calc truncation rate
    hit_or_wait_all_frames = 1 - preds_all # wait=1, hit=0
    trt = tf.reduce_mean(tf.reduce_prod(hit_or_wait_all_frames, [2, 3]), 1)
        # (num thresh,)

    if duration == 1:
        # Forced decision
        preds_last = tf.sign(
            tf.reduce_min(llr_mtx, -1)
            ) + 1
            # 0: wait, 1: hit (one-hot vector)
            # (1, batch, duration=1, num cls) 
        """assert check shape"""
        """assert check all the data points in the batch is are now one-hot vectors."""
        preds_last = tf.tile(preds_last, [num_thresh, 1, 1, 1])
        preds_all_trunc = preds_last 
            # (num thresh, batch, 1, num cls)

        # Calc hitting times
        mht = tf.constant(1., tf.float32)
        vht = tf.constant(0., tf.float32)

        # Calc confusion matrices
        preds = preds_all_trunc[:, :, 0, :]
            # (num thresh, batch, 1, num cls): one-hot vectors

        labels_oh = tf.one_hot(labels_concat, depth=num_classes, axis=1)
            # (batch, num cls)
        labels_oh = tf.expand_dims(labels_oh, axis=0)
        labels_oh = tf.tile(labels_oh, [num_thresh, 1, 1])
            # (num thresh, batch, num cls)

        preds = tf.expand_dims(preds, axis=-2)
        labels_oh = tf.expand_dims(labels_oh, axis=-1)
        confmx = tf.cast(tf.reduce_sum(labels_oh * preds, axis=1), tf.int32)
            # (num thresh, num cls, num cls): label axis x pred axis

    else:
        # Forced decision
        preds_last = tf.sign(
            tf.reduce_min(llr_mtx[:, :, -1, :, :], -1)
            ) + 1
            # 0: wait, 1: hit (one-hot vector)
            # (1, batch, num cls) 
        """assert check shape"""
        """assert check all the data points in the batch is are now one-hot vectors."""
        preds_last = tf.expand_dims(preds_last, 2) 
            # (1, batch, 1, num cls)
        preds_last = tf.tile(preds_last, [num_thresh, 1, 1, 1])
            # (num thresh, batch, 1, num cls)
        preds_all_trunc = tf.concat([preds_all[:, :, :-1, :], preds_last], 2) 
            # (num thresh, batch, duration - 1, num cls)
            # + (num thresh, batch, 1, num cls)
            # = (num thresh, batch, duration, num cls) 
            # Now, preds_all_trunc[i, j, t, :] for fixed i and j is
            # a one-hot vector for t = duration - 1 
            # and
            # filled with 0 or a one-hot vector for t != duration - 1.
        """ assert: check this """

        # Calc mean hitting time
        mask = tf.constant([i+1 for i in range(duration)][::-1], tf.float32)
        mask = tf.tile(mask, [num_thresh * batch_size * num_classes])
        mask = tf.reshape(mask, [num_thresh, batch_size, num_classes, duration])
        mask = tf.transpose(mask, [0, 1, 3, 2])
        masked = preds_all_trunc * mask 
            # (num thresh, batch, duration, num cls)
        hitidx = tf.reduce_max(masked, axis=2)
            # (num thresh, batch, num cls)
        hittimes = duration - tf.reduce_max(hitidx, axis=2) + 1
            # (num thresh, batch)
        mht, vht = tf.nn.moments(hittimes, axes=[1])
            # (num thresh,)
        
        # Calc confusion matrix
        preds = tf.argmax(hitidx, axis=2)
            # (num thresh, batch,)
        preds = tf.one_hot(preds, depth=num_classes, axis=2)
            # (num thresh, batch, num cls)

        labels_oh = tf.one_hot(labels_concat, depth=num_classes, axis=1)
            # (batch, num cls)
        labels_oh = tf.expand_dims(labels_oh, axis=0)
        labels_oh = tf.tile(labels_oh, [num_thresh, 1, 1])
            # (num thresh, batch, num cls)

        preds = tf.expand_dims(preds, axis=-2)
        labels_oh = tf.expand_dims(labels_oh, axis=-1)
        confmx = tf.cast(tf.reduce_sum(labels_oh * preds, axis=1), tf.int32)
            # (num thresh, num cls, num cls): label axis x pred axis

    return confmx, mht, vht, trt


def classwise_MHT(llrs, labels, values, classes, batch_thresh):
    """ Mean hitting times only of some classes.
    Args:
        llrs: (batch, duration, num classes, num classes)
        labels: (batch, )
        values: A Tensor with shape (number of thersholds,)
            Ascending order.
            Typically, `values` is generated like
            >>> start, stop = get_LLR_min_and_max(llrs) 
            >>> values = get_linspace(start, stop, num_thresh, sparsity)
        classes: A list of integers.
        batch_thresh: An int. ~5 is recommended to save GPU memory.
    Returns:
        mht_th: A Tensor with shape (num thresh,).
            If no class data is found in llrs and labels,
            then mht_th = None.
        vht_th: A Tensor with shape (num thresh,).
            If no class data is found in llrs and labels,
            then vht_th = None.
    """
    shapes = llrs.shape
    duration = shapes[1]
    num_classes = shapes[2]
    num_thresh = values.shape[0]

    # Restrict classes (optional)
    llrs_all, labels_all = restrict_classes(llrs, labels, classes)
    if llrs_all is None:
        return None, None

    # Mini-batch threshold processing to save GPU memory 
    #ls_confmx_th = []
    ls_mht_th = []
    ls_vht_th = []
    #ls_trt_th = []
    for i in range(num_thresh // batch_thresh):
        # Generate Thresholds for SPRT
        idx = batch_thresh * i
        itr_values = values[idx : idx + batch_thresh]
        itr_thresh = threshold_generator_with_values(
            itr_values, duration, num_classes) # memory consuming
        thresh_sanity_check(itr_thresh)
        
        # Confusion matrix of SPRT, mean/var hitting time, 
        # and truncation rate
        _, tmp_mht_th, tmp_vht_th, _ = \
            truncated_MSPRT(
                llr_mtx=llrs_all,
                labels_concat=labels_all,
                thresh_mtx=itr_thresh) 
                # Super GPU memory super-consuming if batch_thresh is large.
            # (num thresh, num classes, num classes)
            # (num thresh,)
            # (num thresh,)
            # (num thresh,)
        #ls_confmx_th.append(tmp_confmx_th)
        ls_mht_th.append(tmp_mht_th)
        ls_vht_th.append(tmp_vht_th)
        #ls_trt_th.append(tmp_trt_th)

    #confmx_th = tf.concat(ls_confmx_th, axis=0)
    mht_th = tf.concat(ls_mht_th, axis=0)
    vht_th = tf.concat(ls_vht_th, axis=0)
    #trt_th = tf.concat(ls_trt_th, axis=0)

    return mht_th, vht_th


# Functions: LLR -> confmx
def llr_sequential_confmx(llrs, labels_concat):
    """ For optuna and NP test.
        Calculate the frame-by-frame confusion matrices
        based on the log-likelihood ratios.
    Args:
        llrs: A Tensor 
            with shape (batch, duration, num classes, num classes).
        labels_concat: A non-one-hot label Tensor with shape (batch,). 
            This is the output from 
            datasets.data_processing.sequential_concat(logit_slice, labels_slice).
    Returns:
        seqconfmx_llr: A Tensor with shape (duration, num classes, num classes). 
            The sequential confusion matrices of framewise LLRs with thresh=0.
    """
    llrs_shape = llrs.shape
    duration = llrs_shape[1]
    num_classes = llrs_shape[2]

    # To avoid double hit due to the values exactly equal to zero in LLRs.
    tri = tf.ones_like(llrs) # memory consuming
    triu = tf.linalg.band_part(tri, 0, -1) # Upper triangular part.
    tril = tf.linalg.band_part(tri, -1, 0) # Lower triangular part.
    llrs -= 1e-12 * (triu - tril) 
        # (batch, duration , num cls, num cls) 

    # Forced decision
    preds = tf.sign(
        tf.reduce_min(llrs, 3)
        ) + 1
        # 0: wait, 1: hit (one-hot vector)
        # (batch, duration, num cls) 

    # Calc confusion matrices
    labels_oh = tf.one_hot(labels_concat, depth=num_classes, axis=1)
        # (batch, num cls)
    labels_oh = tf.expand_dims(labels_oh, axis=1)
    labels_oh = tf.tile(labels_oh, [1, duration, 1])
        # (batch, duration, num cls)

    preds = tf.expand_dims(preds, axis=-2)
    labels_oh = tf.expand_dims(labels_oh, axis=-1)
    seqconfmx = tf.cast(tf.reduce_sum(labels_oh * preds, axis=0), tf.int32)
        # (duration, num cls, num cls): label axis x pred axis

    return seqconfmx


# Functions to save GPU memory. Used in plot_SATC_casual.ipynb as of July 7, 2020.
def get_LLR_min_and_max(llrs):
    """
    Args:
        llrs: A Tensor with shape 
            [batch, duration, num classes, num classes].
            Anti-symmetric matrix.
    Returns:
        llr_min: A scalar Tensor. Is min|off-diag(llrs)| (!= 0 is guaranteed).
        llr_max: A scalar Tensor. Is max|llrs| (!= 0 is guaranteed).
    """

    llrs_shape = llrs.shape
    num_classes = llrs_shape[-1]

    # Remove 0 LLRs in off-diags
    tri = tf.ones_like(llrs) # memory consuming
    triu = tf.linalg.band_part(tri, 0, -1) # Upper triangular part.
    tril = tf.linalg.band_part(tri, -1, 0) # Lower triangular part.
    llrs -= 1e-12 * (triu - tril) 

    # Calc non-zero max and min of |LLRs|
    llrs_abs = tf.abs(llrs)
    llrs_max = tf.reduce_max(llrs_abs)
        # max |LLRs|
    tmp = tf.linalg.tensor_diag([1.] * num_classes) * llrs_max
    tmp = tf.reshape(tmp, [1, 1, num_classes, num_classes])
    llrs_min = tf.reduce_min(llrs_abs + tmp)
        # strictly positive (non-zero) minimum of LLRs
    
    return llrs_min, llrs_max


def get_linspace(start, stop, num, sparsity):
    """
    Args:
        start: A float.
        stop: A float.
        num: An integer, the number of points to be generated.
            E.g., if sparsity="linspace",
            1 => thresh = start
            2 => thresh = start, stop
            3 => thresh = start, (start+stop)/2, stop
            ... (linspaced float numbers)...
        sparsity: "linspace", "logspace", "unirandom", or "lograndom".
    Return:
        values: A Tensor with shape
            (num,)
    """
    assert start < stop
    
    # Single-valued threshold matrix
    if sparsity == "linspace":
        thresh = tf.linspace(start, stop, num)    
            # (num,)

    elif sparsity == "logspace":
        thresh = tf.math.exp(
            tf.linspace(
                tf.math.log(start), 
                tf.math.log(stop), 
                num))
            # (num,)

    elif sparsity == "unirandom":
        thresh = tf.random.uniform(shape=[num])
        thresh = tf.sort(thresh)
        thresh = ((stop - start) * thresh) + start
            # (num,). Ascending order

    elif sparsity == "lograndom":
        thresh = tf.random.uniform(shape=[num])
        thresh = tf.sort(thresh)
        thresh = tf.math.exp(
            ((tf.math.log(stop) - tf.math.log(start)) * thresh)\
                + tf.math.log(start))
            # (num,). Ascending order.

    else:
        raise ValueError

    values = thresh
    return values
    
    
def threshold_generator_with_values(values, duration, num_classes):
    """
    Args:
        values: A Tensor with shape (-1,) 
            Values = strictly positive, float thresholds. 
        duration: An int.
        num_classes: An int.
    Returns:
        thresh: A Tensor with shape
            (len(list_values), duration, num_classes, num_classes).
            In each matrix, 
            diag = 0, and off-diag shares a single value > 0.
            Matrices are sorted in ascending order of the values
            w.r.t. axis=0.
    """
    num_thresh = values.shape[0]
    thresh = tf.reshape(values, [num_thresh, 1, 1, 1])
    thresh = tf.tile(thresh, [1, duration, num_classes, num_classes])
        # (num thresh, num cls, num cls)
    mask = tf.linalg.tensor_diag([-1.] * num_classes) + 1
    thresh *= mask
        # Now diag = 0.
    thresh += mask * 1e-11
        # Avoids 0 threholds, which may occur
        # when logits for different classes have the same value,
        # e.g., 0, due to loss of significance.
        # This operation may cause sparsity of SAT curve 
        # if llr_min is << 1e-11, but such a case is ignorable 
        # in practice, according to my own experience. 
    
    return thresh


# Functions: other statistical tests
def NP_test(llrs, labels, length=None):
    """ Neyman-Pearson Test.
    Args:
        llrs: A Tensor 
            with shape (batch, duration, num classes, num classes).
        labels: A Tensor with shape (batch,).
        length: A integer or None.
            If this is None, the NPT result of all frames is returned.
            If this is not None, 
            length must be 1 <= length <= duration, 
            and the NPT result at the length-th frame (1-base) is returned.
    Returns:
        A Tensor with shape (num classes, num classes) if length is not None,
        else (duration, num classes, num classes).
    Remark:
        - Currently, only threshold = 0 is supported.
        - The input `llrs` must be the log-likelihood ratios 
          ---  otherwise this is not the NPT --- even though 
          this function works if `llrs` comprises arbitrary scores.
    """
    if length is None:
        return llr_sequential_confmx(llrs, labels)

    else:
        assert 1 <= length <= llrs.shape[1]
        return llr_sequential_confmx(llrs, labels)[length - 1]