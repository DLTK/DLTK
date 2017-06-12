from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages

from dltk.core.modules import *

class ConvolutionalAutoencoder(AbstractModule):
    """Convolutional Autoencoder

    This module builds a convolutional autoencoder with varying number of layers and hidden units.
    """
    def __init__(self, num_hidden_units=2, filters=(16, 64, 128, 256),
                 strides=((1, 1, 1), (2, 2, 2), (1, 1, 1), (2, 2, 2)), relu_leakiness=0.01,
                 name='conv_ae'):
        """Constructs a convolutional autoencoder

        Parameters
        ----------
        num_hidden_units : int
            number of hidden units
        filters : list or tuple
            filters for convolutional layers
        strides : list or tuple
            strides to be used for convolutions
        relu_leakiness : float
            leakines of relu nonlinearity
        name : string
            name of the network
        """
        self.num_hidden_units = num_hidden_units
        self.filters = filters
        self.strides = strides
        self.relu_leakiness = relu_leakiness
        self.in_filter = None

        assert (len(strides) == len(filters))

        super(ConvolutionalAutoencoder, self).__init__(name)

    def _build(self, inp, is_training=True):
        """Constructs a ResNetFCN using the input tensor

        Parameters
        ----------
        inp : tf.Tensor
            input tensor
        is_training : bool
            flag to specify whether this is training - passed to batch normalization

        Returns
        -------
        dict
            output dictionary containing:
                - `hidden` - hidden units
                - `y_` - reconstruction of the autoencoder

        """

        if self.in_filter is None:
            self.in_filter = inp.get_shape().as_list()[-1]
        assert(self.in_filter == inp.get_shape().as_list()[-1], 'Network was built for a different input shape')

        out = {}

        x = inp

        for scale in range(len(self.filters)):
            with tf.variable_scope('encoder_%d' % (scale)):
                x = Convolution(self.filters[scale], strides=self.strides[scale])(x)
                x = BatchNorm()(x, is_training)
                x = leaky_relu(x, self.relu_leakiness)
                print(x.get_shape().as_list())

        x_shape = x.get_shape().as_list()

        x = tf.reshape(x, (tf.shape(x)[0], np.prod(x_shape[1:])))
        x = Linear(self.num_hidden_units)(x)

        out['hidden'] = x

        x = Linear(np.prod(x_shape[1:]))(x)
        x = tf.reshape(x, [tf.shape(x)[0]] + list(x_shape)[1:])

        for scale in reversed(range(len(self.filters))):
            with tf.variable_scope('decoder_%d' % (scale)):
                f = self.filters[scale - 1] if scale > 0 else self.in_filter
                x = BatchNorm()(x, is_training)
                x = leaky_relu(x, self.relu_leakiness)
                x = TransposedConvolution(f, strides=self.strides[scale])(x)
                print(x.get_shape().as_list())

        out['x_'] = x

        return out