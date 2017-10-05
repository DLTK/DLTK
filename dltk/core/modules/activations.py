from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf

from dltk.core.modules.base import AbstractModule


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

class PReLU(AbstractModule):
    def __init__(self, name='prelu'):
        self._rank = None
        self._shape = None
        super(PReLU, self).__init__(name)

    def _build(self, inp):
        if self._rank is None:
            self._rank = len(inp.get_shape().as_list())

        assert self._rank == len(inp.get_shape().as_list()), 'Module was initilialised for a different input'
        if self._rank > 2:
            if self._shape is None:
                self._shape = [inp.get_shape().as_list()[-1]]
            assert self._shape[0] == inp.get_shape().as_list()[-1], 'Module was initilialised for a different input'
        else:
            self._shape = []

        leakiness = tf.get_variable('leakiness', shape=self._shape, initializer=tf.constant_initializer(0.01),
                                    collections=self.TRAINABLE_COLLECTIONS)
        return tf.maximum(inp, leakiness * inp)