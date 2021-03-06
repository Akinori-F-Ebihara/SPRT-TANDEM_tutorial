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

import tensorflow as tf
import tensorflow_addons as tfa 
import numpy as np

def get_optimizer(learning_rates, decay_steps, name_optimizer, flag_wd, weight_decay=0.):
    """
    Args:
        learning_rates: A list of learning rates.
        decay_steps: A list of steps at which learnig rate decays.
        name_optimizer: A str.
        flag_wd: A boolean.
        weight_decay: A float.
    """
    if not (len(learning_rates) - 1 == len(decay_steps)):
        raise ValueError(
            "len(learning_rates) - 1 == len(decay_steps) must hold. Now: len(learning_rates)={}, len(decay_steps){}".format(len(learning_rates), len(decay_steps))
            )
    learning_rates = [float(v) for v in learning_rates]
    decay_steps = [float(v) for v in decay_steps]
    weight_decay = float(weight_decay)
    flag_wd_in_loss = False

    # Get scheduler
    lr_scheduler = tf.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=decay_steps, 
        values=learning_rates)

    # Get optimizer
    if name_optimizer == "adam":
        if flag_wd:
            optimizer = tfa.optimizers.AdamW(
                weight_decay=weight_decay, 
                learning_rate=lr_scheduler)
        else:
            optimizer = tf.optimizers.Adam(learning_rate=lr_scheduler)

    elif name_optimizer == "momentum":
        if flag_wd:
            optimizer = tfa.optimizers.SGDW(
                weight_decay=weight_decay, 
                learning_rate=lr_scheduler, 
                momentum=0.9, 
                nesterov=True)
        else:
            optimizer = tf.optimizers.SGD(learning_rate=lr_scheduler, momentum=0.9)

    elif name_optimizer == "sgd":
        if flag_wd:
            optimizer = tfa.optimizers.SGDW(weight_decay=weight_decay, learning_rate=lr_scheduler)
        else:
            optimizer = tf.optimizers.SGD(learning_rate=lr_scheduler)

    elif name_optimizer == "rmsprop":
        if flag_wd:
            optimizer = tf.optimizers.RMSprop(learning_rate=lr_scheduler, rho=0.9, momentum=0.0)
            flag_wd_in_loss = True 
        else:
            optimizer = tf.optimizers.RMSprop(learning_rate=lr_scheduler, rho=0.9, momentum=0.0)

    elif name_optimizer == "adagrad":
        if flag_wd:
            optimizer = tf.optimizers.Adagrad(
                learning_rate=lr_scheduler, 
                initial_accumulator_value=0.1)
            flag_wd_in_loss = True 
        else:
            optimizer = tf.optimizers.Adagrad(
                learning_rate=lr_scheduler, 
                initial_accumulator_value=0.1)

    else:
        raise ValueError("Wrong optimizer: {}".format(name_optimizer))

    return optimizer, flag_wd_in_loss


