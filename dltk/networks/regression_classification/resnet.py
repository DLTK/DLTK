from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf

from dltk.core.residual_unit import vanilla_residual_unit_3d


def resnet_3d(inputs,
              num_classes,
              num_res_units=1,
              filters=(16, 32, 64, 128),
              strides=((1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
              mode=tf.estimator.ModeKeys.EVAL,
              use_bias=False,
              activation=tf.nn.relu6,
              kernel_initializer=tf.initializers.variance_scaling(distribution='uniform'),
              bias_initializer=tf.zeros_initializer(),
              kernel_regularizer=None, bias_regularizer=None):
    """
    Regression/classification network based on a flexible resnet
    architecture [1] using residual units proposed in [2]. The downsampling
    of features is done via strided convolutions. On each resolution scale s
    are num_convolutions with filter size = filters[s]. strides[s]
    determine the downsampling factor at each resolution scale.

    [1] K. He et al. Deep residual learning for image recognition. CVPR 2016.
    [2] K. He et al. Identity Mappings in Deep Residual Networks. ECCV 2016.

    Args:
        inputs (tf.Tensor): Input feature tensor to the network (rank 5
            required).
        num_classes (int): Number of output channels or classes.
        num_res_units (int, optional): Number of residual units per resolution
            scale.
        filters (tuple, optional): Number of filters for all residual units at
            each resolution scale.
        strides (tuple, optional): Stride of the first unit on a resolution
            scale.
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
    assert len(strides) == len(filters)
    assert len(inputs.get_shape().as_list()) == 5, \
        'inputs are required to have a rank of 5.'

    relu_op = tf.nn.relu6

    conv_params = {'padding': 'same',
                   'use_bias': use_bias,
                   'kernel_initializer': kernel_initializer,
                   'bias_initializer': bias_initializer,
                   'kernel_regularizer': kernel_regularizer,
                   'bias_regularizer': bias_regularizer}

    x = inputs

    # Inital convolution with filters[0]
    k = [s * 2 if s > 1 else 3 for s in strides[0]]
    x = tf.layers.conv3d(x, filters[0], k, strides[0], **conv_params)
    tf.logging.info('Init conv tensor shape {}'.format(x.get_shape()))

    # Residual feature encoding blocks with num_res_units at different
    # resolution scales res_scales
    res_scales = [x]
    saved_strides = []
    for res_scale in range(1, len(filters)):

        # Features are downsampled via strided convolutions. These are defined
        # in `strides` and subsequently saved
        with tf.variable_scope('unit_{}_0'.format(res_scale)):

            x = vanilla_residual_unit_3d(
                inputs=x,
                out_filters=filters[res_scale],
                strides=strides[res_scale],
                activation=activation,
                mode=mode)
        saved_strides.append(strides[res_scale])

        for i in range(1, num_res_units):

            with tf.variable_scope('unit_{}_{}'.format(res_scale, i)):

                x = vanilla_residual_unit_3d(
                    inputs=x,
                    out_filters=filters[res_scale],
                    strides=(1, 1, 1),
                    activation=activation,
                    mode=mode)
        res_scales.append(x)
        tf.logging.info('Encoder at res_scale {} tensor shape: {}'.format(
            res_scale, x.get_shape()))

    # Global pool and last unit
    with tf.variable_scope('pool'):
        x = tf.layers.batch_normalization(
            x, training=mode == tf.estimator.ModeKeys.TRAIN)
        x = relu_op(x)

        axis = tuple(range(len(x.get_shape().as_list())))[1:-1]
        x = tf.reduce_mean(x, axis=axis, name='global_avg_pool')

        tf.logging.info('Global pool shape {}'.format(x.get_shape()))

    with tf.variable_scope('last'):
        x = tf.layers.dense(inputs=x,
                            units=num_classes,
                            activation=None,
                            use_bias=conv_params['use_bias'],
                            kernel_initializer=conv_params['kernel_initializer'],
                            bias_initializer=conv_params['bias_initializer'],
                            kernel_regularizer=conv_params['kernel_regularizer'],
                            bias_regularizer=conv_params['bias_regularizer'],
                            name='hidden_units')

        tf.logging.info('Output tensor shape {}'.format(x.get_shape()))

    # Define the outputs
    outputs['logits'] = x

    with tf.variable_scope('pred'):

        y_prob = tf.nn.softmax(x)
        outputs['y_prob'] = y_prob

        y_ = tf.argmax(x, axis=-1) \
            if num_classes > 1 \
            else tf.cast(tf.greater_equal(x[..., 0], 0.5), tf.int32)
        outputs['y_'] = y_

    return outputs
