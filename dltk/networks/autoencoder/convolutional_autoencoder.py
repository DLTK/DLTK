from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
import numpy as np


def convolutional_autoencoder_3d(inputs, num_convolutions=1,
                                 num_hidden_units=128, filters=(16, 32, 64),
                                 strides=((2, 2, 2), (2, 2, 2), (2, 2, 2)),
                                 mode=tf.estimator.ModeKeys.TRAIN,
                                 use_bias=False,
                                 activation=tf.nn.relu6,
                                 kernel_initializer=tf.initializers.variance_scaling(distribution='uniform'),
                                 bias_initializer=tf.zeros_initializer(),
                                 kernel_regularizer=None,
                                 bias_regularizer=None):
    """Convolutional autoencoder with num_convolutions on len(filters)
        resolution scales. The downsampling of features is done via strided
        convolutions and upsampling via strided transpose convolutions. On each
        resolution scale s are num_convolutions with filter size = filters[s].
        strides[s] determine the downsampling factor at each resolution scale.

    Args:
        inputs (tf.Tensor): Input tensor to the network, required to be of
            rank 5.
        num_convolutions (int, optional): Number of convolutions per resolution
            scale.
        num_hidden_units (int, optional): Number of hidden units.
        filters (tuple or list, optional): Number of filters for all
            convolutions at each resolution scale.
        strides (tuple or list, optional): Stride of the first convolution on a
            resolution scale.
        mode (str, optional): One of the tf.estimator.ModeKeys strings: TRAIN,
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
    assert len(strides) == len(filters)
    assert len(inputs.get_shape().as_list()) == 5, \
        'inputs are required to have a rank of 5.'

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
    for res_scale in range(0, len(filters)):

        for i in range(0, num_convolutions - 1):
            with tf.variable_scope('enc_unit_{}_{}'.format(res_scale, i)):
                x = conv_op(inputs=x,
                            filters=filters[res_scale],
                            kernel_size=(3, 3, 3),
                            strides=(1, 1, 1),
                            **conv_params)

                x = tf.layers.batch_normalization(
                    inputs=x,
                    training=mode == tf.estimator.ModeKeys.TRAIN)
                x = activation(x)
                tf.logging.info('Encoder at res_scale {} shape: {}'.format(
                    res_scale, x.get_shape()))

        # Employ strided convolutions to downsample
        with tf.variable_scope('enc_unit_{}_{}'.format(
                res_scale,
                num_convolutions)):

            # Adjust the strided conv kernel size to prevent losing information
            k_size = [s * 2 if s > 1 else 3 for s in strides[res_scale]]

            x = conv_op(inputs=x,
                        filters=filters[res_scale],
                        kernel_size=k_size,
                        strides=strides[res_scale],
                        **conv_params)

            x = tf.layers.batch_normalization(
                x, training=mode == tf.estimator.ModeKeys.TRAIN)
            x = activation(x)
            tf.logging.info('Encoder at res_scale {} tensor shape: {}'.format(
                res_scale, x.get_shape()))

    # Densely connected layer of hidden units
    x_shape = x.get_shape().as_list()
    x = tf.reshape(x, (tf.shape(x)[0], np.prod(x_shape[1:])))

    x = tf.layers.dense(inputs=x,
                        units=num_hidden_units,
                        use_bias=conv_params['use_bias'],
                        kernel_initializer=conv_params['kernel_initializer'],
                        bias_initializer=conv_params['bias_initializer'],
                        kernel_regularizer=conv_params['kernel_regularizer'],
                        bias_regularizer=conv_params['bias_regularizer'],
                        name='hidden_units')

    outputs['hidden_units'] = x
    tf.logging.info('Hidden units tensor shape: {}'.format(x.get_shape()))

    x = tf.layers.dense(inputs=x,
                        units=np.prod(x_shape[1:]),
                        activation=activation,
                        use_bias=conv_params['use_bias'],
                        kernel_initializer=conv_params['kernel_initializer'],
                        bias_initializer=conv_params['bias_initializer'],
                        kernel_regularizer=conv_params['kernel_regularizer'],
                        bias_regularizer=conv_params['bias_regularizer'])

    x = tf.reshape(x, [tf.shape(x)[0]] + list(x_shape)[1:])
    tf.logging.info('Decoder input tensor shape: {}'.format(x.get_shape()))

    # Decoding blocks with num_convolutions at different resolution scales
    # res_scales
    for res_scale in reversed(range(0, len(filters))):

        # Employ strided transpose convolutions to upsample
        with tf.variable_scope('dec_unit_{}_0'.format(res_scale)):

            # Adjust the strided tp conv kernel size to prevent losing
            # information
            k_size = [s * 2 if s > 1 else 3 for s in strides[res_scale]]

            x = tp_conv_op(inputs=x,
                           filters=filters[res_scale],
                           kernel_size=k_size,
                           strides=strides[res_scale],
                           **conv_params)

            x = tf.layers.batch_normalization(
                x, training=mode == tf.estimator.ModeKeys.TRAIN)
            x = activation(x)
            tf.logging.info('Decoder at res_scale {} tensor shape: {}'.format(
                res_scale, x.get_shape()))

        for i in range(1, num_convolutions):
            with tf.variable_scope('dec_unit_{}_{}'.format(res_scale, i)):

                x = conv_op(inputs=x,
                            filters=filters[res_scale],
                            kernel_size=(3, 3, 3),
                            strides=(1, 1, 1),
                            **conv_params)

                x = tf.layers.batch_normalization(
                    x, training=mode == tf.estimator.ModeKeys.TRAIN)
                x = activation(x)
            tf.logging.info('Decoder at res_scale {} tensor shape: {}'.format(
                res_scale, x.get_shape()))

    # A final convolution reduces the number of output features to those of
    # the inputs
    x = conv_op(inputs=x,
                filters=inputs.get_shape().as_list()[-1],
                kernel_size=(1, 1, 1),
                strides=(1, 1, 1),
                **conv_params)

    tf.logging.info('Output tensor shape: {}'.format(x.get_shape()))
    outputs['x_'] = x

    return outputs
