from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf


def simple_super_resolution_3d(inputs,
                               num_convolutions=1,
                               filters=(16, 32, 64),
                               upsampling_factor=(2, 2, 2),
                               mode=tf.estimator.ModeKeys.EVAL,
                               use_bias=False,
                               activation=tf.nn.relu6,
                               kernel_initializer=tf.initializers.variance_scaling(distribution='uniform'),
                               bias_initializer=tf.zeros_initializer(),
                               kernel_regularizer=None,
                               bias_regularizer=None):
    """Simple super resolution network with num_convolutions per feature
        extraction block. Each convolution in a block b has a filter size
        of filters[b].

    Args:
        inputs (tf.Tensor): Input feature tensor to the network (rank 5
            required).
        num_convolutions (int, optional): Number of convolutions.
        filters (tuple, optional): filters (tuple, optional): Number of filters.
        upsampling_factor (tuple, optional): Upsampling factor of the low
            resolution to the high resolution image.
        mode (TYPE, optional): One of the tf.estimator.ModeKeys strings: TRAIN,
            EVAL or PREDICT
        use_bias (bool, optional): Boolean, whether the layer uses a bias.
        activation (optional): A function to use as activation function.
        kernel_initializer (TYPE, optional): An initializer for the convolution
            kernel.
        bias_initializer (TYPE, optional): An initializer for the bias vector.
            If None, no bias will be applied.
        kernel_regularizer (None, optional): Optional regularizer for the
            convolution kernel.
        bias_regularizer (None, optional): Optional regularizer for the bias
            vector.

    Returns:
        dict: dictionary of output tensors

    """

    outputs = {}
    assert len(inputs.get_shape().as_list()) == 5, \
        'inputs are required to have a rank of 5.'
    assert len(upsampling_factor) == 3, \
        'upsampling factor is required to be of length 3.'

    conv_op = tf.layers.conv3d
    tp_conv_op = tf.layers.conv3d_transpose

    conv_params = {'padding': 'same',
                   'use_bias': use_bias,
                   'kernel_initializer': kernel_initializer,
                   'bias_initializer': bias_initializer,
                   'kernel_regularizer': kernel_regularizer,
                   'bias_regularizer': bias_regularizer}

    x = inputs
    tf.logging.info('Input tensor shape {}'.format(x.get_shape()))

    # Convolutional feature encoding blocks with num_convolutions at different
    # resolution scales res_scales
    for unit in range(0, len(filters)):

        for i in range(0, num_convolutions):

            with tf.variable_scope('enc_unit_{}_{}'.format(unit, i)):

                x = conv_op(inputs=x,
                            filters=filters[unit],
                            kernel_size=(3, 3, 3),
                            strides=(1, 1, 1),
                            **conv_params)

                x = tf.layers.batch_normalization(
                    x, training=mode == tf.estimator.ModeKeys.TRAIN)

                x = activation(x)

                tf.logging.info('Encoder at unit_{}_{} tensor '
                                'shape: {}'.format(unit, i, x.get_shape()))

    # Upsampling
    with tf.variable_scope('upsampling_unit'):

        # Adjust the strided tp conv kernel size to prevent losing information
        k_size = [u * 2 for u in upsampling_factor]
        x = tp_conv_op(inputs=x,
                       filters=inputs.get_shape().as_list()[-1],
                       kernel_size=k_size,
                       strides=upsampling_factor,
                       **conv_params)

    tf.logging.info('Output tensor shape: {}'.format(x.get_shape()))
    outputs['x_'] = x

    return outputs
