from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import numpy as np
from dltk.core.upsample import linear_upsample_3d


def dcgan_generator_3d(inputs,
                       out_filters,
                       num_convolutions=1,
                       filters=(16, 32, 64),
                       strides=((2, 2, 2), (2, 2, 2), (2, 2, 2)),
                       mode=tf.estimator.ModeKeys.TRAIN, use_bias=False):
    """Deep convolutional generative adversial network (DCGAN) generator
        network. with num_convolutions on len(filters) resolution scales. The
        upsampling of features is done via strided transpose convolutions. On
        each resolution scale s are num_convolutions with filter size =
        filters[s]. strides[s] determine the upsampling factor at each
        resolution scale.

    Args:
        inputs (tf.Tensor): Input noise tensor to the network.
        out_filters (int): Number of output filters.
        num_convolutions (int, optional): Number of convolutions per resolution
            scale.
        filters (tuple, optional): Number of filters for all convolutions at
            each resolution scale.
        strides (tuple, optional): Stride of the first convolution on a
            resolution scale.
        mode (TYPE, optional): One of the tf.estimator.ModeKeys strings: TRAIN,
            EVAL or PREDICT
        use_bias (bool, optional): Boolean, whether the layer uses a bias.

    Returns:
        dict: dictionary of output tensors
    """
    outputs = {}
    assert len(strides) == len(filters)
    assert len(inputs.get_shape().as_list()) == 5, \
        'inputs are required to have a rank of 5.'

    conv_op = tf.layers.conv3d
    relu_op = tf.nn.relu6
    upsample_op = linear_upsample_3d

    conv_params = {'padding': 'same',
                   'use_bias': use_bias,
                   'kernel_initializer': tf.uniform_unit_scaling_initializer(),
                   'bias_initializer': tf.zeros_initializer(),
                   'kernel_regularizer': None,
                   'bias_regularizer': None}

    x = inputs
    tf.logging.info('Input tensor shape {}'.format(x.get_shape()))

    for res_scale in range(0, len(filters)):
        for i in range(0, num_convolutions - 1):

            with tf.variable_scope('gen_unit_{}_{}'.format(res_scale, i)):

                x = conv_op(inputs=x,
                            filters=filters[res_scale],
                            kernel_size=(3, 3, 3),
                            strides=(1, 1, 1),
                            **conv_params)

                x = tf.layers.batch_normalization(
                    x, training=mode == tf.estimator.ModeKeys.TRAIN)

                x = relu_op(x)

        with tf.variable_scope('gen_unit_{}_{}'.format(
                res_scale, num_convolutions)):

            tf.logging.info('Generator at res_scale before up {} tensor '
                            'shape: {}'.format(res_scale, x.get_shape()))

            x = upsample_op(inputs=x,
                            strides=strides[res_scale],
                            use_bias=use_bias)

            tf.logging.info('Generator at res_scale after up {} tensor '
                            'shape: {}'.format(res_scale, x.get_shape()))

            x = tf.layers.batch_normalization(
                x, training=mode == tf.estimator.ModeKeys.TRAIN)

            x = relu_op(x)
            tf.logging.info('Generator at res_scale {} tensor shape: '
                            '{}'.format(res_scale, x.get_shape()))

    for i in range(0, num_convolutions - 1):

        with tf.variable_scope('gen_unit_{}_{}'.format(len(filters), i)):

            x = conv_op(inputs=x,
                        filters=filters[-1],
                        kernel_size=(3, 3, 3),
                        strides=(1, 1, 1),
                        **conv_params)

            x = tf.layers.batch_normalization(
                x, training=mode == tf.estimator.ModeKeys.TRAIN)

            x = relu_op(x)

    x = conv_op(inputs=x,
                filters=out_filters,
                kernel_size=(3, 3, 3),  #TODO  IS THIS CORRECT?
                strides=(1, 1, 1), **conv_params)

    outputs['gen'] = x

    return outputs


def dcgan_discriminator_3d(inputs,
                           num_convolutions=1,
                           filters=(16, 32, 64),
                           strides=((2, 2, 2), (2, 2, 2), (2, 2, 2)),
                           mode=tf.estimator.ModeKeys.EVAL,
                           use_bias=False):
    """Deep convolutional generative adversarial network (DCGAN) discriminator
        network with num_convolutions on len(filters) resolution scales. The
        downsampling of features is done via strided convolutions. On each
        resolution scale s are num_convolutions with filter size = filters[s].
         strides[s] determine the downsampling factor at each resolution scale.

    Args:
        inputs (tf.Tensor): Input tensor to the network, required to be of
            rank 5.
        num_convolutions (int, optional): Number of convolutions per resolution
            scale.
        filters (tuple, optional): Number of filters for all convolutions at
            each resolution scale.
        strides (tuple, optional): Stride of the first convolution on a
            resolution scale.
        mode (TYPE, optional): One of the tf.estimator.ModeKeys strings: TRAIN,
            EVAL or PREDICT.
        use_bias (bool, optional): Boolean, whether the layer uses a bias.

    Returns:
        dict: dictionary of output tensors
    """
    outputs = {}
    assert len(strides) == len(filters)
    assert len(inputs.get_shape().as_list()) == 5,\
        'inputs are required to have a rank of 5.'

    conv_op = tf.layers.conv3d
    relu_op = tf.nn.relu6

    conv_params = {'padding': 'same',
                   'use_bias': use_bias,
                   'kernel_initializer': tf.uniform_unit_scaling_initializer(),
                   'bias_initializer': tf.zeros_initializer(),
                   'kernel_regularizer': None,
                   'bias_regularizer': None}

    x = inputs
    tf.logging.info('Input tensor shape {}'.format(x.get_shape()))

    for res_scale in range(0, len(filters)):
        for i in range(0, num_convolutions - 1):

            with tf.variable_scope('disc_unit_{}_{}'.format(res_scale, i)):

                x = conv_op(inputs=x,
                            filters=filters[res_scale],
                            kernel_size=(3, 3, 3),
                            strides=(1, 1, 1),
                            **conv_params)

                x = relu_op(x)

        # Employ strided convolutions to downsample
        with tf.variable_scope('disc_unit_{}_{}'.format(
                res_scale, num_convolutions)):

            # Adjust the strided conv kernel size to prevent losing information
            k_size = [s * 2 if s > 1 else 3 for s in strides[res_scale]]

            x = conv_op(inputs=x,
                        filters=filters[res_scale],
                        kernel_size=k_size,
                        strides=strides[res_scale],
                        **conv_params)

            x = relu_op(x)
            tf.logging.info('Discriminator at res_scale {} tensor shape: '
                            '{}'.format(res_scale, x.get_shape()))

    x_shape = x.get_shape().as_list()
    x = tf.reshape(x, (tf.shape(x)[0], np.prod(x_shape[1:])))

    x = tf.layers.dense(inputs=x,
                        filters=1,
                        use_bias=True,
                        kernel_initializer=conv_params['kernel_initializer'],
                        bias_initializer=conv_params['bias_initializer'],
                        kernel_regularizer=conv_params['kernel_regularizer'],
                        bias_regularizer=conv_params['bias_regularizer'],
                        name='out')

    outputs['logits'] = x

    outputs['probs'] = tf.nn.sigmoid(x)

    outputs['pred'] = tf.cast((x > 0.5), tf.int32)

    return outputs
