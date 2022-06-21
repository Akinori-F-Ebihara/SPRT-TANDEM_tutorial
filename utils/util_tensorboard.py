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

"""Ref: https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514"""
import os
from io import BytesIO  # Python 3.x
import numpy as np
import tensorflow as tf
import scipy.misc

class TensorboardLogger():
    """ 
    Usage:
        # Example code (TF2.0.0)
        global_step = np.array(0, dtype=np.int64)
        tf.summary.experimental.set_step(global_step)
        tblogger = TensorboardLogger(logdir)
        with tblogger.writer.as_default():
            tblogger.scalar_summary(tab, value, description)
    """
    def __init__(self, root_tblogs, subproject_name, exp_phase, comment, time_stamp):
        """Create a summary writer logging to root_tblogs + naming rule shown below.
        Args:
            root_tblogs: A string. 
            subproject_name: A string.
            comment: A string.
            time_stamp: A str of a time stamp. e.g., time_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]
        Remark:
            Tensorboard logs of one run will be saved in "root_tblogs/subproject_name_exp_phase/comment_time_stamp"            """

        # Naming Rule
        self.root_tblogs = root_tblogs
        self.subproject_name = subproject_name
        self.exp_phase = exp_phase
        self.comment = comment
        self.time_stamp = time_stamp
        self.dir_tblogs = self.root_tblogs + "/" + self.subproject_name + "_" + self.exp_phase + "/" + self.comment + "_"+ self.time_stamp
        if not os.path.exists(self.dir_tblogs):
            os.makedirs(self.dir_tblogs)
        print("Set Tensorboard directory: ", self.dir_tblogs)

        # Create a summary writer
        self.writer = tf.summary.create_file_writer(self.dir_tblogs, flush_millis=10000)

    def scalar_summary(self, tag, value, step=None, description=None):
        """Log a scalar variable.
           Invoke in writer.as_default() context."""
        tf.summary.scalar(name=tag, data=value, step=step, description=description)

    def histo_summary(self, tag, values, step=None, 
        buckets=None, description=None):
        """Log a histogram of the tensor of values.
           Invoke in writer.as_default() context."""
        tf.summary.histogram(name=tag, data=values, 
            step=step, buckets=None, description=description)

    # Under construction.
    def image_summary(self, tag, images, step): # to be updated for TF2
        """Log a list of images.
           Invoke in writer.as_default() context."""
        
        img_summaries = []
        for _, img in enumerate(images):
            # Write the image to a string
            s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")
            
            # Create an Image object
            img_sum = tf.compat.v1.Summary.Image(
                h=img.shape[1]
            )
            # Create a Summary value
            img_summaries.append(tf.summary.Value(tag="{tag}/{i}", image=img_sum))
            
        # Create and write Summary
        summary = tf.compat.v1.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)



