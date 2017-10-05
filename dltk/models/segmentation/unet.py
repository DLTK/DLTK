from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages

from dltk.core.modules import *


class UpsampleAndConcat(AbstractModule):
    """UNET upsampling module according to O. Ronneberger.

    """
    def __init__(self, strides, name='upandconcat'):
        """Constructs an UpsampleAndConcat module

        Parameters
        ----------
        strides : list or tuple
            strides to use for upsampling
        name : string
            name of the module
        """
        self.strides = strides
        super(UpsampleAndConcat, self).__init__(name)

    def _build(self, x, x_up):
        """Applies the UpsampleAndConcat operation

        Parameters
        ----------
        x : tf.Tensor
            tensor to be upsampled
        x_up : tf.Tensor
            tensor from the same scale to be convolved and added to the upsampled tensor

        Returns
        -------
        tf.Tensor
            output of the operation
        """

        t_conv = BilinearUpsample(strides=self.strides)(x)

        return tf.concat(axis=-1,values=[x_up, t_conv])


class ResUNET(SaveableModule):
    """ ResUNET module with residual encoder

    This module builds a UNET for segmentation using a residual encoder.
    """

    output_keys = ['logits', 'y_prob', 'y_']

    def __init__(self, num_classes, num_residual_units=3, filters=(16, 64, 128, 256, 512),
                 strides=((1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2), (1, 1, 1)), relu_leakiness=0.1,
                 name='resnetfcn'):
        """Builds a residual UNET for segmentation

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
        self.input_filters = None
        self.rank = None
        super(ResUNET, self).__init__(name)

    def _build_input_placeholder(self):
        """Abstract function to build input placeholders
        """

        assert self.input_filters is not None, 'self.input_filters must be defined'

        self.input_placeholders = [tf.placeholder(tf.float32, shape=[None, ] * (1 + len(self.strides[0]))
                                                                   + [self.input_filters])]

    def _build(self, inp, is_training=True):
        """Constructs a ResNetUNET using the input tensor

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

        assert len(strides) == len(filters)

        if self.input_filters is None:
            self.input_filters = inp.get_shape().as_list()[-1]
            self._build_input_placeholder()
        assert self.input_filters == inp.get_shape().as_list()[-1]

        if self.rank is None:
            self.rank = len(strides[0])
        assert len(inp.get_shape().as_list()) == self.rank + 2, \
            'Stride gives rank {} input is rank {}'.format(self.rank, len(inp.get_shape().as_list()) - 2)


        x = inp

        x = Convolution(filters[0], strides=strides[0])(x)
        tf.logging.info(x.get_shape())

        # residual feature encoding blocks with num_residual_units at different scales defined via strides
        scales = [x]
        saved_strides = []
        for scale in range(1, len(filters)):
            with tf.variable_scope('unit_%d_0' % (scale)):
                x = VanillaResidualUnit(filters[scale], stride=strides[scale])(x, is_training=is_training)
            saved_strides.append(strides[scale])
            for i in range(1, self.num_residual_units):
                with tf.variable_scope('unit_%d_%d' % (scale, i)):
                    x = VanillaResidualUnit(filters[scale], stride=[1] * self.rank)(x, is_training=is_training)
            scales.append(x)
            tf.logging.info('feat_scale_%d shape %s', scale, x.get_shape())

        # decoder
        for scale in range(len(filters) - 2, -1, -1):
            with tf.variable_scope('upsample_%d' % scale):
                tf.logging.info('Building upsampling for scale %d with x (%s) x_up (%s) stride (%s)'
                                % (scale, x.get_shape().as_list(), scales[scale].get_shape().as_list(),
                                   saved_strides[scale]))
                x = UpsampleAndConcat(saved_strides[scale])(x, scales[scale])
            with tf.variable_scope('up_unit_%d_0' % (scale)):
                x = VanillaResidualUnit(filters[scale], stride=[1] * self.rank)(x, is_training=is_training)
            tf.logging.info('up_%d shape %s', scale, x.get_shape())

        with tf.variable_scope('last'):
            x = Convolution(self.num_classes, 1, strides=[1] * self.rank)(x)

        outputs['logits'] = x
        tf.logging.info('last conv shape %s', x.get_shape())

        with tf.variable_scope('pred'):
            y_prob = tf.nn.softmax(x)
            outputs['y_prob'] = y_prob
            y_ = tf.argmax(x, axis=-1) if self.num_classes > 1 else tf.cast(tf.greater_equal(x[..., 0], 0.5), tf.int32)
            outputs['y_'] = y_

        return outputs
    