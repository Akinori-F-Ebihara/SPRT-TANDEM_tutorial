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

# Create NMNIST-H before starting the training.
# See /data-directory/create_UCF101.

from __future__ import absolute_import, division, print_function
import datetime, sys
import numpy as np
import tensorflow as tf

from datasets.data_processing import read_tfrecords_UCF101,\
    which_dataset_UCF101
from models.backbones_ti import LSTMModelLite
from models.optimizers import get_optimizer
from models.losses import get_gradient_lstm
from utils.misc import load_yaml, set_gpu_devices, fix_random_seed
from utils.util_tensorboard import TensorboardLogger
from utils.performance_metrics import llr_sequential_confmx,\
    seqconfmx_to_metricsV2,\
    truncated_MSPRT,\
    calc_llrs, calc_oblivious_llrs, threshold_generator, thresh_sanity_check
from utils.util_optuna import run_optuna, suggest_parameters_CSL
from utils.util_ckpt import checkpoint_logger

# Load Params
config_path = "./configs/config_ti_UCF101.yaml"
config = load_yaml(config_path)

#============================================
if config["param_llr_loss"] == 0:
    config["list_lllr"] = [0.]
if (config["flag_prefactor"] == False) and\
    (config["flag_prior_ratio"] == False) and\
    (config["version"] != "F"):
    config["list_beta"] = [-1] 
#============================================

# GPU settings
set_gpu_devices(config["gpu"])

# Set Random Seeds (Optional)
fix_random_seed(flag_seed=config["flag_seed"], seed=config["seed"])


# Subfunctions
def tblog_writer_train(tblogger, losses, global_step, 
    thrshconfmx, mhttr, vhttr, trttr, thresh_mtx, 
    dict_metrics_llr):
    # Losses
    tblogger.scalar_summary("train_loss/Sum_loss", 
        losses[1] + losses[2], int(global_step))
    tblogger.scalar_summary("train_loss/Multiplet_loss",
        losses[1], int(global_step))
    tblogger.scalar_summary("train_loss/LLLR", 
        losses[2], int(global_step))

    # Mean macro-averaged recall and accuracy
    mean_macRecall = tf.reduce_mean(
        dict_metrics_llr["SNS"][:, -2])
    tblogger.scalar_summary("train_metric_LLR/mean_macroRecall", 
        mean_macRecall, int(global_step))
    mean_exJAC = tf.reduce_mean(
        dict_metrics_llr["exJAC"])
    tblogger.scalar_summary("train_metric_LLR/mean_Accuracy", 
        mean_exJAC, int(global_step))


    # SPRT: metrics, MHT, VHT, Trunc, MaxThresh
    cnt = 0
    for mht, vht, trt, thresh in zip(
        mhttr.numpy(), vhttr.numpy(), 
        trttr.numpy(), thresh_mtx.numpy()):
        cnt += 1
        tblogger.scalar_summary(
            "train_metric_SPRT/MHT_{}th_thresh".format(cnt), 
            mht, int(global_step))
        tblogger.scalar_summary(
            "train_metric_SPRT/VHT_{}th_thresh".format(cnt), 
            vht, int(global_step))
        tblogger.scalar_summary(
            "train_metric_SPRT/Trunc_{}th_thresh".format(cnt),
            trt, int(global_step))
        tblogger.scalar_summary(
            "train_LLR_scale/{}th {} LLR component".format(cnt, config["sparsity"]), 
            np.max(thresh), int(global_step))
    list_metrics_sprt = seqconfmx_to_metricsV2(thrshconfmx)["SNS"]
    for i, v in enumerate(list_metrics_sprt):
        tblogger.scalar_summary(
            "train_metric_SPRT/MacroRecall {}th thresh".format(i + 1), 
            v[-2], int(global_step))

    # Framewise metrics
    tblogger.scalar_summary("train_metric_LLR/macroRecall_frame001", 
        dict_metrics_llr["SNS"][0, -2], int(global_step)) 
    tblogger.scalar_summary("train_metric_LLR/macroRecall_frame015",
        dict_metrics_llr["SNS"][14, -2], int(global_step)) 
    tblogger.scalar_summary("train_metric_LLR/macroRecall_frameMAX", 
        dict_metrics_llr["SNS"][-1, -2], int(global_step)) 


def tblog_writer_val(tblogger, global_step, losses_val,\
    dict_metrics_llr_val, wd_reg):
    # Lossse
    tblogger.scalar_summary("valid_loss/Sum_loss", 
        losses_val[1] + losses_val[2], int(global_step))
    tblogger.scalar_summary("valid_loss/Multiplet_loss", 
        losses_val[1], int(global_step))
    tblogger.scalar_summary("valid_loss/LLLR", 
        losses_val[2], int(global_step))      
    tblogger.scalar_summary("weight_decay/weight_decay", 
        wd_reg, int(global_step))  

    # Mean macro-averaged recall and accuracy
    mean_macRecall_val = tf.reduce_mean(
        dict_metrics_llr_val["SNS"][:, -2])
    tblogger.scalar_summary("valid_metric_LLR/mean_macroRecall", 
        mean_macRecall_val, int(global_step))
    mean_exJAC_val = tf.reduce_mean(
        dict_metrics_llr_val["exJAC"])
    tblogger.scalar_summary("valid_metric_LLR/mean_Accuracy",  
        mean_exJAC_val, int(global_step))

    # Framewise
    tblogger.scalar_summary(
        "valid_metric_LLR/macroRecall_frame001", 
        dict_metrics_llr_val["SNS"][0, -2], int(global_step))   
    tblogger.scalar_summary(
        "valid_metric_LLR/macroRecall_frame015", 
        dict_metrics_llr_val["SNS"][14, -2], int(global_step))            
    tblogger.scalar_summary(
        "valid_metric_LLR/macroRecall_frameMAX", 
        dict_metrics_llr_val["SNS"][-1, -2], int(global_step))


def validation_loop(dataset_va, model, num_validset):
    # Validation loop
    for iter_bv, feats in enumerate(dataset_va):
        cnt = iter_bv + 1
        x_batch = feats[0]
        x_batch = tf.reshape(x_batch, (-1, config["duration"], config["feat_dim"])) 
            # (batch, duration, feat dims)
        y_batch = feats[1]
            # (batch, )

        # Calc loss, confmx, and mean hitting time 
        if iter_bv == 0:
            # Calc loss
            _, losses, logits_concat = get_gradient_lstm( 
                model, x_batch, y_batch, 
                training=False, order_sprt=config["order_sprt"],
                duration=config["duration"], oblivious=config["oblivious"],
                version=config["version"], flag_wd=False, flag_mgn=False, calc_grad=False, 
                param_multiplet_loss=1., param_llr_loss=1., param_wd=0.,
                flag_prefactor=config["flag_prefactor"], 
                classwise_sample_sizes=config["classwise_sample_sizes_UCF101-{}".format(config["duration"])], 
                beta=config["beta"],
                flag_prior_ratio=config["flag_prior_ratio"])

            # LLR
            if config["oblivious"]:
                llrsval = calc_oblivious_llrs(logits_concat)
            else:
                llrsval = calc_llrs(logits_concat)
            seqconfmx_llr = llr_sequential_confmx(llrsval, y_batch)
                # (duration, num cls, num cls)

        else:
            _, losses_tmp, logits_concat = get_gradient_lstm(
                model, x_batch, y_batch, 
                training=False, order_sprt=config["order_sprt"],
                duration=config["duration"], oblivious=config["oblivious"],
                version=config["version"], flag_wd=False, flag_mgn=False, calc_grad=False, 
                param_multiplet_loss=1., param_llr_loss=1., param_wd=0.,
                flag_prefactor=config["flag_prefactor"], 
                classwise_sample_sizes=config["classwise_sample_sizes_UCF101-{}".format(config["duration"])], 
                beta=config["beta"],
                flag_prior_ratio=config["flag_prior_ratio"])

            for iter_idx in range(len(losses)):
                losses[iter_idx] += losses_tmp[iter_idx]

            # LLR
            if config["oblivious"]:
                llrsval = calc_oblivious_llrs(logits_concat)
            else:
                llrsval = calc_llrs(logits_concat)
            seqconfmx_llr += llr_sequential_confmx(llrsval, y_batch)
                # (duration, num cls, num cls)

        # Verbose
        if ((iter_bv + 1) % 10 == 0) or (iter_bv == 0):
            sys.stdout.write(
                "\rValidation Iter: {:3d}/{:3d}".format(
                    iter_bv + 1, 
                    (num_validset // config["batch_size"]) + 1 \
                    if num_validset % config["batch_size"] != 0 \
                    else num_validset // config["batch_size"])
                )
            sys.stdout.flush()
   
    print("")

    # Losses
    for iter_idx in range(len(losses)):
        losses[iter_idx] /= cnt

    # Weight decay
    wd_reg = 0.
    for variables in model.trainable_variables:
        wd_reg += tf.nn.l2_loss(variables)

    dict_metrics_llr = seqconfmx_to_metricsV2(
        seqconfmx_llr)

    return losses, dict_metrics_llr,\
        wd_reg


# Main Function
def objective(trial):
    # Timestamp and assertion
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]
    assert (config["exp_phase"] == "tuning") or (config["exp_phase"] == "stat")\
        or (config["exp_phase"] == "try")

    # Suggest parameters if necessary
    ####################################
    if config["exp_phase"] == "tuning":
        list_suggest = suggest_parameters_CSL(
            trial, 
            list_lr=config["list_lr"], 
            list_bs=config["list_bs"], 
            list_opt=config["list_opt"], 
            list_do=config["list_do"], 
            list_wd=config["list_wd"],
            list_lllr=config["list_lllr"],
            list_order=config["list_order"],
            list_beta=config["list_beta"]) 

        print("##############################################################")
        print("Suggest params: ", list_suggest)
        print("##############################################################")

        learning_rate = list_suggest[0]
        batch_size = list_suggest[1]
        name_optimizer = list_suggest[2]
        dropout = list_suggest[3]
        weight_decay = list_suggest[4]
        param_llr_loss = list_suggest[5]
        order_sprt = list_suggest[6]
        beta = list_suggest[7]

        config["learning_rates"] = [learning_rate, learning_rate*0.1]
        config["batch_size"] = batch_size
        config["name_optimizer"] = name_optimizer
        config["dropout"] = dropout
        config["weight_decay"] = weight_decay
        config["param_llr_loss"] = param_llr_loss
        config["order_sprt"] = order_sprt
        config["beta"] = beta 

    assert (config["param_llr_loss"] != 0) or (config["param_multiplet_loss"] != 0)

    # Load data
    ##################################
    assert np.any([config["HxW"] == [240, 320], config["HxW"] == [256, 256]]),\
        "config['HxW'] = {}".format(config["HxW"])
    assert np.any(config["duration"] == np.array([50, 150]))
    assert config["duration"] > config["order_sprt"]
    key_tr, key_va, key_te, key_numtr, key_numva, key_numte \
        = which_dataset_UCF101(config["duration"], config["HxW"][0], 
        config["HxW"][1], config["split"])
    num_trainset = config[key_numtr]
    num_validset = config[key_numva]
    _ = config[key_numte]
    tfr_tr = config[key_tr]
    tfr_va = config[key_va]
    tfr_te = config[key_te]
    dataset_tr, dataset_va,\
        _ = \
        read_tfrecords_UCF101(
            record_file_tr=tfr_tr, 
            record_file_va=tfr_va,
            record_file_te=tfr_te,
            batch_size=config["batch_size"], 
            flag_shuffle=config["flag_shuffle"],
            shuffle_buffer_size=10000)
        
    # Model
    ######################################
    model = LSTMModelLite(
        config["num_classes"], 
        config["width_lstm"], 
        dropout=config["dropout"], 
        activation=config["activation"])

    # Get optimizer
    optimizer, flag_wd_in_loss = get_optimizer(
        learning_rates=config["learning_rates"], 
        decay_steps=config["decay_steps"], 
        name_optimizer=config["name_optimizer"], 
        flag_wd=config["flag_wd"], 
        weight_decay=config["weight_decay"])        


    # Tensorboard and checkpoints
    ####################################
    # Define global step
    global_step = tf.Variable(0, name="global_step", dtype=tf.int32)

    # Checkpoint
    _, ckpt_manager = checkpoint_logger(
        global_step, 
        model, 
        optimizer, 
        config["flag_resume"], 
        config["root_ckptlogs"], 
        config["subproject_name"], 
        config["exp_phase"],
        config["comment"], 
        now, 
        config["path_resume"], 
        config["max_to_keep"],
        config_path)

    # Tensorboard
    tblogger = TensorboardLogger(
        root_tblogs=config["root_tblogs"], 
        subproject_name=config["subproject_name"], 
        exp_phase=config["exp_phase"], 
        comment=config["comment"], 
        time_stamp=now)


    # Training
    ####################################
    # Start training
    with tblogger.writer.as_default():
        # Initialization
        best = 0.

        # Training and validation
        num_epochs = (config["num_iter"] * config["batch_size"]) // num_trainset\
            if (config["num_iter"] * config["batch_size"]) % num_trainset == 0\
            else (config["num_iter"] * config["batch_size"]) // num_trainset + 1

        for epoch in range(num_epochs):
            # Training loop
            for iter_b, feats in enumerate(dataset_tr):
                x_batch = feats[0]
                x_batch = tf.reshape(x_batch, (-1, config["duration"], config["feat_dim"])) 
                    # (batch, duration, feat dims)
                y_batch = feats[1]
                    # (batch, )

                # Show summary of model
                if (epoch == 0) and (iter_b == 0):
                    model.build(input_shape=x_batch.shape)
                    model.summary() 

                # Calc loss and grad, and backpropagation
                grads, losses, logits_concat = get_gradient_lstm(
                    model, 
                    x_batch, 
                    y_batch, 
                    training=True, 
                    order_sprt=config["order_sprt"],
                    duration=config["duration"],
                    oblivious=config["oblivious"],
                    version=config["version"],
                    param_multiplet_loss=config["param_multiplet_loss"], 
                    param_llr_loss=config["param_llr_loss"], 
                    param_wd=config["weight_decay"], 
                    flag_wd=flag_wd_in_loss,
                    flag_mgn=config["flag_mgn"],
                    calc_grad=True,
                    flag_prefactor=config["flag_prefactor"], 
                    classwise_sample_sizes=config["classwise_sample_sizes_UCF101-{}".format(config["duration"])], 
                    beta=config["beta"],
                    flag_prior_ratio=config["flag_prior_ratio"])
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                global_step.assign_add(1)
                
                #Verbose
                if tf.equal(global_step % config["train_display_step"], 0) \
                    or tf.equal(global_step, 1):
                    print("Global Step={:7d}/{:7d} Epoch={:4d}/{:4d} Iter={:5d}/{:5d}: sum loss={:7.5f}, multiplet loss={:7.5f}, LLLR={:7.5f}".format(
                        int(global_step),
                        config["num_iter"],
                        epoch + 1, 
                        num_epochs, 
                        iter_b + 1, 
                        (num_trainset // config["batch_size"]) + 1\
                            if num_trainset % config["batch_size"] != 0\
                            else num_trainset // config["batch_size"], 
                        losses[1]+losses[2], 
                        losses[1], 
                        losses[2]))

                    # Confusion matrix of SPRT and mean hitting time of a batch
                    if config["oblivious"]:
                        llrs = calc_oblivious_llrs(logits_concat)
                    else:
                        llrs = calc_llrs(logits_concat)
                    thresh_mtx = threshold_generator(llrs, config["num_thresh"], config["sparsity"])
                    thresh_sanity_check(thresh_mtx)
                    thrshconfmx, mht, vht, trt = \
                        truncated_MSPRT(
                            llr_mtx=llrs,
                            labels_concat=y_batch,
                            thresh_mtx=thresh_mtx)

                    # Confusion matrix of LLR at every frame
                    seqconfmx_llr = llr_sequential_confmx(llrs, y_batch)
                    dict_metrics_llr = seqconfmx_to_metricsV2(seqconfmx_llr)

                    # Tensorboard
                    tblog_writer_train(
                        tblogger,
                        losses, 
                        global_step,
                        thrshconfmx,
                        mht, vht, trt,
                        thresh_mtx,
                        dict_metrics_llr)

                # Validation
                #################################
                if tf.equal(global_step % config["valid_step"], 0) or\
                    tf.equal(global_step, 1):

                    losses_val, dict_metrics_llr_val, wd_reg =\
                    validation_loop(
                        dataset_va, 
                        model,
                        num_validset)

                    # Tensorboard for validation
                    tblog_writer_val(
                        tblogger, global_step, losses_val, 
                        dict_metrics_llr_val, 
                        wd_reg)

                    # For exp_phase="tuning", optuna
                    mean_macRecall_val = tf.reduce_mean(
                        dict_metrics_llr_val["SNS"][:,-2])
                    print("Temporal-mean of Macro-averaged Recall on validation: {}".format(mean_macRecall_val))

                    if best < mean_macRecall_val:
                        best = mean_macRecall_val
                    
                        # Save checkpoint
                        ckpt_manager._checkpoint_prefix = \
                            ckpt_manager._checkpoint_prefix[:ckpt_manager._checkpoint_prefix.rfind("/") + 1] + \
                            "ckpt_step{}_macrec{:.5f}".format(int(global_step), best)
                        save_path_prefix = ckpt_manager.save()
                        print("Best value updated. Saved checkpoint for step {}: {}".format(
                            int(global_step), save_path_prefix))

                if tf.equal(global_step, config["num_iter"]):
                    break

            if tf.equal(global_step, config["num_iter"]):
                break

    # Final processes
    ###############################################
    # Return best valid result for optuna
    return 1 - best


if __name__ == '__main__':
    run_optuna(config["root_dblogs"], config["subproject_name"], 
        config["exp_phase"], objective, config["num_trials"])

