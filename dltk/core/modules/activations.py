from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf


def leaky_relu(x, leakiness):
    """ Leaky RELU

    Parameters
    ----------
    x : tf.Tensor
        input tensor
    leakiness : float
        leakiness of RELU

    Returns
    -------
    tf.Tensor
        Tensor with applied leaky RELU

    """
    return tf.maximum(x, leakiness * x)