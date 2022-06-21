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

import optuna
import sys, os

def run_optuna(root_dblogs, subproject_name, exp_phase, objective, nb_trials):
    """ Load or create study and optimize objective.
    Args:
        root_dblogs: A str. The root directory for .db files.
        sugproject_name: A string. Used for study name and storage name.
        exp_phase: A string. Used for study name and storage name.
    """
    # Paths
    study_name = subproject_name + "_" + exp_phase
    storage_name = "sqlite:///" + root_dblogs + "/" + study_name + ".db"
    if not os.path.exists(root_dblogs):
        os.makedirs(root_dblogs)

    # Load or create study, and start optimization
    study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True)
    study.optimize(objective, n_trials=nb_trials)


def suggest_parameters(trial, 
    list_lr, list_bs, list_opt, list_do, 
    list_wd, list_lllr, list_order):
    """ Suggest hyperparameters.
    Args:
        trial: A trial object for optuna optimization.
        list_lr: A list of floats. Candidates of learning rates.
        list_bs: A list of ints. Candidates of batch sizes.
        list_opt: A list of strings. Candidates of optimizers.
        list_do: dropout
        list_wd: weight decay
    Returns:
        learning_rate: A float.
        batch_size: An int.
        name_optimizer: A string.
    """
    # load yaml interprrets, e.g., 1e-2 as a string...
    for iter_idx in range(len(list_lr)):
        list_lr[iter_idx] = float(list_lr[iter_idx])

    learning_rate = trial.suggest_categorical('learning_rate', list_lr)
    batch_size = trial.suggest_categorical('batch_size', list_bs)
    name_optimizer = trial.suggest_categorical('optimizer', list_opt)
    dropout = trial.suggest_categorical('dropout', list_do)
    weight_decay = trial.suggest_categorical('weight_decay', list_wd)
    param_llr_loss = trial.suggest_categorical('param_llr_loss', list_lllr)
    order_sprt = trial.suggest_categorical('order_sprt', list_order)

    return learning_rate, batch_size, name_optimizer, dropout,\
        weight_decay, param_llr_loss, order_sprt


def suggest_parameters_CSL(trial, 
    list_lr, list_bs, list_opt, list_do, 
    list_wd, list_lllr, list_order, list_beta):
    """ Suggest hyperparameters.
    Args:
        trial: A trial object for optuna optimization.
        list_lr: A list of floats. Candidates of learning rates.
        list_bs: A list of ints. Candidates of batch sizes.
        list_opt: A list of strings. Candidates of optimizers.
        list_do: dropout
        list_wd: weight decay
        list_beta: beta for cost-sensitive learning
    Returns:
        learning_rate: A float.
        batch_size: An int.
        name_optimizer: A string.
    """
    # load yaml interprrets, e.g., 1e-2 as a string...
    for iter_idx in range(len(list_lr)):
        list_lr[iter_idx] = float(list_lr[iter_idx])

    learning_rate = trial.suggest_categorical('learning_rate', list_lr)
    batch_size = trial.suggest_categorical('batch_size', list_bs)
    name_optimizer = trial.suggest_categorical('optimizer', list_opt)
    dropout = trial.suggest_categorical('dropout', list_do)
    weight_decay = trial.suggest_categorical('weight_decay', list_wd)
    param_llr_loss = trial.suggest_categorical('param_llr_loss', list_lllr)
    order_sprt = trial.suggest_categorical('order_sprt', list_order)
    beta = trial.suggest_categorical('beta', list_beta)

    return learning_rate, batch_size, name_optimizer, dropout,\
        weight_decay, param_llr_loss, order_sprt, beta


def suggest_parameters_DRE_UCF101(trial, 
    list_lr, list_bs, list_opt, 
    list_wd, list_multLam, list_order, list_beta):
    """ Suggest hyperparameters.
    Args:
        trial: A trial object for optuna optimization.
        list_lr: A list of floats. Candidates of learning rates.
        list_bs: A list of ints. Candidates of batch sizes.
        list_opt: A list of strings. Candidates of optimizers.
        list_wd: A list of floats. weight decay
        list_multLam: A list of floats. Prefactor of the second term of BARR.
        list_order: A list of integers. Order of SPRT-TANDEM.
        list_beta: A list of floats. Beta for cost-sensitive learning.
    Returns:
        learning_rate: A float.
        batch_size: An int.
        name_optimizer: A string.
        weight_decay: A float.
        param_multLam: A float.
        order_sprt: An int.
        beta: A float.
    """
    # load yaml interprrets, e.g., 1e-2 as a string...
    for iter_idx in range(len(list_lr)):
        list_lr[iter_idx] = float(list_lr[iter_idx])

    learning_rate = trial.suggest_categorical('learning_rate', list_lr)
    batch_size = trial.suggest_categorical('batch_size', list_bs)
    name_optimizer = trial.suggest_categorical('optimizer', list_opt)
    weight_decay = trial.suggest_categorical('weight_decay', list_wd)
    param_multLam = trial.suggest_categorical('param_multLam', list_multLam)
    order_sprt = trial.suggest_categorical('order_sprt', list_order)
    beta = trial.suggest_categorical('beta', list_beta)

    return learning_rate, batch_size, name_optimizer,\
        weight_decay, param_multLam, order_sprt, beta


def suggest_parameters_DRE_NMNIST(trial, 
    list_lr, list_bs, list_opt, 
    list_wd, list_multLam, list_order):
    """ Suggest hyperparameters.
    Args:
        trial: A trial object for optuna optimization.
        list_lr: A list of floats. Candidates of learning rates.
        list_bs: A list of ints. Candidates of batch sizes.
        list_opt: A list of strings. Candidates of optimizers.
        list_wd: A list of floats. weight decay
        list_multLam: A list of floats. Prefactor of the second term of BARR.
        list_order: A list of integers. Order of SPRT-TANDEM.
    Returns:
        learning_rate: A float.
        batch_size: An int.
        name_optimizer: A string.
        weight_decay: A float.
        param_multLam: A float.
        order_sprt: An int.
    """
    # load yaml interprrets, e.g., 1e-2 as a string...
    for iter_idx in range(len(list_lr)):
        list_lr[iter_idx] = float(list_lr[iter_idx])

    learning_rate = trial.suggest_categorical('learning_rate', list_lr)
    batch_size = trial.suggest_categorical('batch_size', list_bs)
    name_optimizer = trial.suggest_categorical('optimizer', list_opt)
    weight_decay = trial.suggest_categorical('weight_decay', list_wd)
    param_multLam = trial.suggest_categorical('param_multLam', list_multLam)
    order_sprt = trial.suggest_categorical('order_sprt', list_order)

    return learning_rate, batch_size, name_optimizer,\
        weight_decay, param_multLam, order_sprt


def suggest_parameters_fe(trial, list_lr, list_bs, list_opt, list_do, list_wd):
    """ Suggest hyperparameters.
    Args:
        trial: A trial object for optuna optimization.
        list_lr: A list of floats. Candidates of learning rates.
        list_bs: A list of ints. Candidates of batch sizes.
        list_opt: A list of strings. Candidates of optimizers.
        list_do: dropout
        list_wd: weight decay 
    Returns:
        learning_rate: A float.
        batch_size: An int.
        name_optimizer: A string.
        ...
    """
    # load yaml interprrets 1e-2 as string
    for iter_idx in range(len(list_lr)):
        list_lr[iter_idx] = float(list_lr[iter_idx])

    learning_rate = trial.suggest_categorical('learning_rate', list_lr)
    batch_size = trial.suggest_categorical('batch_size', list_bs)
    name_optimizer = trial.suggest_categorical('optimizer', list_opt)
    dropout = trial.suggest_categorical('dropout', list_do)
    weight_decay = trial.suggest_categorical('weight_decay', list_wd)

    return learning_rate, batch_size, name_optimizer, dropout, weight_decay

