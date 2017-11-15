from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import numpy as np

def prelu(x, alpha_initializer=tf.constant_initializer()):
    alpha = tf.get_variable('alpha', shape=[], dtype=tf.float32,
                            initializer=alpha_initializer)
    
    return leaky_relu(x, alpha)

def leaky_relu(x, alpha=0.01):
    return tf.maximum(x, alpha * x)