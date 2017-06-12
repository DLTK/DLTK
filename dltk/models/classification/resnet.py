from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages

from dltk.core.modules import *

class ResNet(AbstractModule):
    """ResNet module

    This module builds a ResNet for classification according to He et al. 2015
    """
    def __init__(self, num_classes, num_residual_units=5, filters=[16, 16, 32, 64],
                 strides=[[1, 1, 1], [1, 1, 1], [2, 2, 2], [2, 2, 2]], relu_leakiness=0.01,
                 name='resnet32'):
        """Builds a residual FCN for segmentation

        Parameters
        ----------
        num_classes : int
            number of classes to segment
        num_residual_units : int
            number of residual units per scale
        filters : tuple or list
            number of filters per scale. The first is used for the initial convolution without residual connections
        strides : tuple or list
            strides per scale. The first is used for the initial convolution without residual connections
        relu_leakiness : float
            leakiness of the relus used
        name : string
            name of the network
        """
        self.num_classes = num_classes
        self.num_residual_units = num_residual_units
        self.filters = filters
        self.strides = strides
        self.relu_leakiness = relu_leakiness
        super(ResNet, self).__init__(name)

    def _build(self, inp, is_training=True):
        """Constructs a ResNet using the input tensor

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
                - `logits` - logits of the classification
                - `y_prob` - classification probabilities
                - `y_` - prediction of the classification

        """
        outputs = {}
        filters = self.filters
        strides = self.strides

        assert (len(strides) == len(filters))

        x = inp

        x = Convolution(filters[0], strides=strides[0])(x)
        print(x.get_shape())

        for scale in range(1, len(filters)):
            with tf.variable_scope('unit_%d_0' % (scale)):
                x = VanillaResidualUnit(filters[scale], stride=strides[scale])(x, is_training=is_training)
            for i in range(1, self.num_residual_units):
                with tf.variable_scope('unit_%d_%d' % (scale, i)):
                    x = VanillaResidualUnit(filters[scale],
                                            stride=[1,] * len(strides[scale]))(x, is_training=is_training)
            tf.logging.info('feat_scale_%d shape %s', scale, x.get_shape())
            print(x.get_shape())

        with tf.variable_scope('unit_last'):
            x = BatchNorm()(x)
            x = leaky_relu(x, self.relu_leakiness)
            axis = tuple(range(len(x.get_shape().as_list())))[1:-1]
            print(axis)
            x = tf.reduce_mean(x, axis=axis, name='global_avg_pool')
            print(x.get_shape())

        with tf.variable_scope('logits'):
            x = tf.reshape(x, (tf.shape(x)[0], filters[-1]))
            x = Linear(self.num_classes)(x)

        outputs['logits'] = x
        tf.logging.info('last conv shape %s', x.get_shape())

        with tf.variable_scope('pred'):
            y_prob = tf.nn.softmax(x)
            outputs['y_prob'] = y_prob
            y_ = tf.argmax(x, axis=-1)
            outputs['y_'] = y_

        return outputs