from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf


def dense_unit_3d(inputs,
                  out_filters,
                  growing_rate,
                  num_convolutions=2,
                  kernel_size=(3, 3, 3),
                  mode=tf.estimator.ModeKeys.EVAL,
                  use_bias=False,
                  activation=tf.nn.leaky_relu,
                  kernel_initializer=tf.initializers.variance_scaling(distribution='uniform'),
                  bias_initializer=tf.zeros_initializer(),
                  kernel_regularizer=None,
                  bias_regularizer=None):
    """Implementation of a 3D residual unit according to [1]. This
        implementation supports strided convolutions and automatically
        handles different input and output filters.

    [1] G. Huang et al. Densely connected convolutional networks. CVPR 2017.

    Args:
        inputs (tf.Tensor): Input tensor to the residual unit. Is required to
            have a rank of 5 (i.e. [batch, x, y, z, channels]).
        out_filters (int): Number of convolutional filters to be produced.
        growing_rate (int): Number of convolutional filters to grow by.
        num_convolutions (int): Number of convolutions in dense unit.
        kernel_size (tuple, optional): Size of the convoltional kernels
            used in the sub units
        mode (str, optional): One of the tf.estimator.ModeKeys: TRAIN, EVAL or
            PREDICT
        activation (optional): A function to use as activation function.
        use_bias (bool, optional): Train a bias with each convolution.
        kernel_initializer (TYPE, optional): Initialisation of convolution kernels
        bias_initializer (TYPE, optional): Initialisation of bias
        kernel_regularizer (None, optional): Additional regularisation op
        bias_regularizer (None, optional): Additional regularisation op

    Returns:
        tf.Tensor: Output of the residual unit
    """

    conv_params = {'padding': 'same',
                   'use_bias': use_bias,
                   'kernel_initializer': kernel_initializer,
                   'bias_initializer': bias_initializer,
                   'kernel_regularizer': kernel_regularizer,
                   'bias_regularizer': bias_regularizer}

    in_filters = inputs.get_shape().as_list()[-1]
    assert in_filters == inputs.get_shape().as_list()[-1], \
        'Module was initialised for a different input shape'

    x = inputs
    previous_x = [x]

    # Sub feature extraction units
    for i in range(num_convolutions):
        with tf.variable_scope('sub_unit_{}'.format(i)):
            current_input = tf.concat(previous_x, 4)

            x = tf.layers.batch_normalization(
                current_input, training=mode == tf.estimator.ModeKeys.TRAIN)
            x = activation(x)

            x = tf.layers.conv3d(
                inputs=x,
                filters=growing_rate,
                kernel_size=kernel_size,
                **conv_params)

            previous_x += [x]

    x = tf.concat(previous_x, 4)

    x = tf.layers.conv3d(inputs=x, filters=out_filters, kernel_size=1)

    return x
