from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.python.training import moving_averages

from dltk.core.modules.base import AbstractModule


class BatchNorm(AbstractModule):
    """Batch normalization module.

    This module normalises the input tensor with statistics across all but the last dimension. During training
    an exponential moving average is kept to be used during test time.

    """
    def __init__(self, offset=True, scale=True, decay_rate=0.99, eps=1e-3, name='bn'):
        """Constructs the batch normalization module

        Parameters
        ----------
        offset : bool, optional
            flag to toggle the offset part of the additional affine transformation (beta)
        scale : bool, optional
            flag to toggle the scale part of the additional affine transformation (gamma)
        decay_rate : float, optional
            decay of the moving averages
        eps : float, optional
            epsilon added to the variance during normalization
        name : string, optional
            name of this module
        """
        self.offset = offset
        self.scale = scale
        self.decay_rate = decay_rate
        self.eps = eps

        self.param_shape = None
        self.axis = None

        super(BatchNorm, self).__init__(name=name)

    def _build(self, inp, is_training=True, test_local_stats=False):
        """Applies the batch norm operation to an input tensor

        Parameters
        ----------
        inp : tf.Tensor
            input tensor for this module
        is_training : bool, optional
            flag to specify whether this is training. If so, batch statistics are used and the moving averages
            are updated
        test_local_stats : bool, optional
            flag to use batch statistics during test time

        Returns
        -------
        tf.Tensor
            normalized tensor

        """

        if self.param_shape is None:
            self.param_shape = inp.get_shape().as_list()[-1]
        assert(self.param_shape == inp.get_shape().as_list()[-1],
               'Input shape must match parameter shape - was initialised for another shape')

        if self.axis is None:
            self.axis = list(np.arange(len(inp.get_shape().as_list()) - 1))
        assert (len(self.axis) == len(inp.get_shape().as_list()) - 1,
                'Input shape must match axis - was initialised for another shape')

        use_batch_stats = is_training | test_local_stats

        self._beta = tf.get_variable('beta', self.param_shape, tf.float32,
                                     initializer=tf.zeros_initializer(),
                                     collections=self.TRAINABLE_COLLECTIONS) if self.offset else None
        self._gamma = tf.get_variable('gamma', self.param_shape, tf.float32,
                                     initializer=tf.ones_initializer(),
                                      collections=self.TRAINABLE_COLLECTIONS) if self.offset else None

        if self.offset:
            self.variables.append(self._beta)
        if self.scale:
            self.variables.append(self._gamma)

        self._mm = tf.get_variable('moving_mean', self.param_shape, tf.float32,
                                   initializer=tf.zeros_initializer(), trainable=False,
                                   collections=self.MOVING_COLLECTIONS)
        self._mv = tf.get_variable('moving_variance', self.param_shape, tf.float32,
                                   initializer=tf.ones_initializer(), trainable=False,
                                   collections=self.MOVING_COLLECTIONS)

        if use_batch_stats:
            mean, variance = tf.nn.moments(inp, self.axis, name='moments')

            # fix for negative variances - see https://github.com/tensorflow/tensorflow/issues/3290
            variance = tf.maximum(variance, tf.constant(0.))

            if is_training:
                update_mean_op = moving_averages.assign_moving_average(
                    variable=self._mm,
                    value=mean,
                    decay=self.decay_rate,
                    zero_debias=False,
                    name="update_moving_mean").op
                update_variance_op = moving_averages.assign_moving_average(
                    variable=self._mv,
                    value=variance,
                    decay=self.decay_rate,
                    zero_debias=False,
                    name="update_moving_variance").op

                with tf.control_dependencies([update_mean_op, update_variance_op]):
                    mean = tf.identity(mean)
                    variance = tf.identity(variance)
        else:
            mean = tf.identity(self._mm)
            variance = tf.identity(self._mv)

        outp = tf.nn.batch_normalization(inp, mean, variance, self._beta, self._gamma, self.eps, name="bn")

        return outp