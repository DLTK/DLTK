from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf


def prelu(inputs, alpha_initializer=tf.constant_initializer()):
    """Probabilistic ReLu activation function

    Args:
        (tf.Tensor): input Tensor
        alpha_initializer (float, optional): an initial value for alpha

    Returns:
        tf.Tensor: a PreLu activated tensor
    """

    alpha = tf.get_variable('alpha',
                            shape=[],
                            dtype=tf.float32,
                            initializer=alpha_initializer)

    return leaky_relu(inputs, alpha)


def leaky_relu(inputs, alpha=0.1):
    """Leaky ReLu activation function

    Args:
        inputs (tf.Tensor): input Tensor
        alpha (float): leakiness parameter

    Returns:
        tf.Tensor: a leaky ReLu activated tensor
    """
    return tf.maximum(inputs, alpha * inputs)
