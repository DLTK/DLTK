from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf

from dltk.core.residual_unit import vanilla_residual_unit_3d
from dltk.core.upsample import linear_upsample_3d
from dltk.core.activations import leaky_relu


def upsample_and_concat(inputs, inputs2, strides=(2, 2, 2)):
    """Upsampling and concatenation layer according to [1].

    [1] O. Ronneberger et al. U-Net: Convolutional Networks for Biomedical Image
        Segmentation. MICCAI 2015.

    Args:
        inputs (TYPE): Input features to be upsampled.
        inputs2 (TYPE): Higher resolution features from the encoder to
            concatenate.
        strides (tuple, optional): Upsampling factor for a strided transpose
            convolution.

    Returns:
        tf.Tensor: Upsampled feature tensor
    """
    assert len(inputs.get_shape().as_list()) == 5, \
        'inputs are required to have a rank of 5.'
    assert len(inputs.get_shape().as_list()) == len(inputs2.get_shape().as_list()), \
        'Ranks of input and input2 differ'

    # Upsample inputs
    inputs = linear_upsample_3d(inputs, strides)

    return tf.concat(axis=-1, values=[inputs2, inputs])


def residual_unet_3d(inputs,
                     num_classes,
                     num_res_units=1,
                     filters=(16, 32, 64, 128),
                     strides=((1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
                     mode=tf.estimator.ModeKeys.EVAL,
                     use_bias=False,
                     activation=leaky_relu,
                     kernel_initializer=tf.initializers.variance_scaling(distribution='uniform'),
                     bias_initializer=tf.zeros_initializer(),
                     kernel_regularizer=None,
                     bias_regularizer=None):
    """
    Image segmentation network based on a flexible UNET architecture [1]
    using residual units [2] as feature extractors. Downsampling and
    upsampling of features is done via strided convolutions and transpose
    convolutions, respectively. On each resolution scale s are
    num_residual_units with filter size = filters[s]. strides[s] determine
    the downsampling factor at each resolution scale.

    [1] O. Ronneberger et al. U-Net: Convolutional Networks for Biomedical Image
        Segmentation. MICCAI 2015.
    [2] K. He et al. Identity Mappings in Deep Residual Networks. ECCV 2016.

    Args:
        inputs (tf.Tensor): Input feature tensor to the network (rank 5
            required).
        num_classes (int): Number of output classes.
        num_res_units (int, optional): Number of residual units at each
            resolution scale.
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

    conv_params = {'padding': 'same',
                   'use_bias': use_bias,
                   'kernel_initializer': kernel_initializer,
                   'bias_initializer': bias_initializer,
                   'kernel_regularizer': kernel_regularizer,
                   'bias_regularizer': bias_regularizer}

    x = inputs

    # Initial convolution with filters[0]
    x = tf.layers.conv3d(inputs=x,
                         filters=filters[0],
                         kernel_size=(3, 3, 3),
                         strides=strides[0],
                         **conv_params)

    tf.logging.info('Init conv tensor shape {}'.format(x.get_shape()))

    # Residual feature encoding blocks with num_res_units at different
    # resolution scales res_scales
    res_scales = [x]
    saved_strides = []
    for res_scale in range(1, len(filters)):

        # Features are downsampled via strided convolutions. These are defined
        # in `strides` and subsequently saved
        with tf.variable_scope('enc_unit_{}_0'.format(res_scale)):

            x = vanilla_residual_unit_3d(
                inputs=x,
                out_filters=filters[res_scale],
                strides=strides[res_scale],
                activation=activation,
                mode=mode)
        saved_strides.append(strides[res_scale])

        for i in range(1, num_res_units):

            with tf.variable_scope('enc_unit_{}_{}'.format(res_scale, i)):

                x = vanilla_residual_unit_3d(
                    inputs=x,
                    out_filters=filters[res_scale],
                    strides=(1, 1, 1),
                    activation=activation,
                    mode=mode)
        res_scales.append(x)

        tf.logging.info('Encoder at res_scale {} tensor shape: {}'.format(
            res_scale, x.get_shape()))

    # Upsample and concat layers [1] reconstruct the predictions to higher
    # resolution scales
    for res_scale in range(len(filters) - 2, -1, -1):

        with tf.variable_scope('up_concat_{}'.format(res_scale)):

            x = upsample_and_concat(
                inputs=x,
                inputs2=res_scales[res_scale],
                strides=saved_strides[res_scale])

        for i in range(0, num_res_units):

            with tf.variable_scope('dec_unit_{}_{}'.format(res_scale, i)):

                x = vanilla_residual_unit_3d(
                    inputs=x,
                    out_filters=filters[res_scale],
                    strides=(1, 1, 1),
                    mode=mode)
        tf.logging.info('Decoder at res_scale {} tensor shape: {}'.format(
            res_scale, x.get_shape()))

    # Last convolution
    with tf.variable_scope('last'):

        x = tf.layers.conv3d(inputs=x,
                             filters=num_classes,
                             kernel_size=(1, 1, 1),
                             strides=(1, 1, 1),
                             **conv_params)

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


def asymmetric_residual_unet_3d(
        inputs,
        num_classes,
        num_res_units=1,
        filters=(16, 32, 64, 128),
        strides=((1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
        mode=tf.estimator.ModeKeys.EVAL,
        use_bias=False,
        activation=leaky_relu,
        kernel_initializer=tf.initializers.variance_scaling(distribution='uniform'),
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None):
    """
    Image segmentation network based on a flexible UNET architecture [1]
    using residual units [2] as feature extractors. Downsampling and
    upsampling of features is done via strided convolutions and transpose
    convolutions, respectively. On each resolution scale s are
    num_residual_units with filter size = filters[s]. strides[s] determine
    the downsampling factor at each resolution scale.

    [1] O. Ronneberger et al. U-Net: Convolutional Networks for Biomedical Image
        Segmentation. MICCAI 2015.
    [2] K. He et al. Identity Mappings in Deep Residual Networks. ECCV 2016.

    Args:
        inputs (tf.Tensor): Input feature tensor to the network (rank 5
            required).
        num_classes (int): Number of output classes.
        num_res_units (int, optional): Number of residual units at each
            resolution scale.
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

    conv_params = {'padding': 'same',
                   'use_bias': use_bias,
                   'kernel_initializer': kernel_initializer,
                   'bias_initializer': bias_initializer,
                   'kernel_regularizer': kernel_regularizer,
                   'bias_regularizer': bias_regularizer}

    x = inputs

    # Initial convolution with filters[0]
    x = tf.layers.conv3d(inputs=x,
                         filters=filters[0],
                         kernel_size=(3, 3, 3),
                         strides=strides[0],
                         **conv_params)

    tf.logging.info('Init conv tensor shape {}'.format(x.get_shape()))

    # Residual feature encoding blocks with num_res_units at different
    # resolution scales res_scales
    res_scales = [x]
    saved_strides = []
    for res_scale in range(1, len(filters)):

        # Features are downsampled via strided convolutions. These are defined
        # in `strides` and subsequently saved
        with tf.variable_scope('enc_unit_{}_0'.format(res_scale)):

            x = vanilla_residual_unit_3d(
                inputs=x,
                out_filters=filters[res_scale],
                strides=strides[res_scale],
                activation=activation,
                mode=mode)
        saved_strides.append(strides[res_scale])

        for i in range(1, num_res_units):

            with tf.variable_scope('enc_unit_{}_{}'.format(res_scale, i)):

                x = vanilla_residual_unit_3d(
                    inputs=x,
                    out_filters=filters[res_scale],
                    strides=(1, 1, 1),
                    activation=activation,
                    mode=mode)
        res_scales.append(x)

        tf.logging.info('Encoder at res_scale {} tensor shape: {}'.format(
            res_scale, x.get_shape()))

    # Upsample and concat layers [1] reconstruct the predictions to higher
    # resolution scales
    for res_scale in range(len(filters) - 2, -1, -1):

        with tf.variable_scope('up_concat_{}'.format(res_scale)):

            x = upsample_and_concat(
                inputs=x,
                inputs2=res_scales[res_scale],
                strides=saved_strides[res_scale])

        with tf.variable_scope('dec_unit_{}'.format(res_scale)):

            x = vanilla_residual_unit_3d(
                inputs=x,
                out_filters=filters[res_scale],
                strides=(1, 1, 1),
                activation=activation,
                mode=mode)
        tf.logging.info('Decoder at res_scale {} tensor shape: {}'.format(
            res_scale, x.get_shape()))

    # Last convolution
    with tf.variable_scope('last'):

        x = tf.layers.conv3d(inputs=x,
                             filters=num_classes,
                             kernel_size=(1, 1, 1),
                             strides=(1, 1, 1),
                             **conv_params)

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
