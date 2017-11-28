from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf

from dltk.core.residual_unit import vanilla_residual_unit_3d
from dltk.core.upsample import linear_upsample_3d


def upscore_layer_3d(inputs,
                     inputs2,
                     out_filters,
                     in_filters=None,
                     strides=(2, 2, 2),
                     mode=tf.estimator.ModeKeys.EVAL,
                     use_bias=False,
                     kernel_initializer=tf.initializers.variance_scaling(distribution='uniform'),
                     bias_initializer=tf.zeros_initializer(),
                     kernel_regularizer=None,
                     bias_regularizer=None):
    """Upscore layer according to [1].

    [1] J. Long et al. Fully convolutional networks for semantic segmentation.
    CVPR 2015.

    Args:
        inputs (tf.Tensor): Input features to be upscored.
        inputs2 (tf.Tensor): Higher resolution features from the encoder to add.
            out_filters (int): Number of output filters (typically, number of
            segmentation classes)
        in_filters (None, optional): None or number of input filters.
        strides (tuple, optional): Upsampling factor for a strided transpose
            convolution.
        mode (TYPE, optional): One of the tf.estimator.ModeKeys strings: TRAIN,
            EVAL or PREDICT
        use_bias (bool, optional): Boolean, whether the layer uses a bias.
        kernel_initializer (TYPE, optional): An initializer for the convolution
            kernel.
        bias_initializer (TYPE, optional): An initializer for the bias vector.
            If None, no bias will be applied.
        kernel_regularizer (None, optional): Optional regularizer for the
            convolution kernel.
        bias_regularizer (None, optional): Optional regularizer for the bias
            vector.

    Returns:
        tf.Tensor: Upscore tensor

    """
    conv_params = {'use_bias': use_bias,
                   'kernel_initializer': kernel_initializer,
                   'bias_initializer': bias_initializer,
                   'kernel_regularizer': kernel_regularizer,
                   'bias_regularizer': bias_regularizer}

    # Compute an upsampling shape dynamically from the input tensor. Input
    # filters are required to be static.
    if in_filters is None:
        in_filters = inputs.get_shape().as_list()[-1]

    assert len(inputs.get_shape().as_list()) == 5, \
        'inputs are required to have a rank of 5.'
    assert len(inputs.get_shape().as_list()) == len(inputs2.get_shape().as_list()), \
        'Ranks of input and input2 differ'

    # Account for differences in the number of input and output filters
    if in_filters != out_filters:
        x = tf.layers.conv3d(inputs=inputs,
                             filters=out_filters,
                             kernel_size=(1, 1, 1),
                             strides=(1, 1, 1),
                             padding='same',
                             name='filter_conversion',
                             **conv_params)
    else:
        x = inputs

    # Upsample inputs
    x = linear_upsample_3d(inputs=x, strides=strides)

    # Skip connection
    x2 = tf.layers.conv3d(inputs=inputs2,
                          filters=out_filters,
                          kernel_size=(1, 1, 1),
                          strides=(1, 1, 1),
                          padding='same',
                          **conv_params)

    x2 = tf.layers.batch_normalization(
        x2, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Return the element-wise sum
    return tf.add(x, x2)


def residual_fcn_3d(inputs,
                    num_classes,
                    num_res_units=1,
                    filters=(16, 32, 64, 128),
                    strides=((1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
                    mode=tf.estimator.ModeKeys.EVAL,
                    use_bias=False,
                    activation=tf.nn.relu6,
                    kernel_initializer=tf.initializers.variance_scaling(distribution='uniform'),
                    bias_initializer=tf.zeros_initializer(),
                    kernel_regularizer=None,
                    bias_regularizer=None):
    """
    Image segmentation network based on an FCN architecture [1] using
    residual units [2] as feature extractors. Downsampling and upsampling
    of features is done via strided convolutions and transpose convolutions,
    respectively. On each resolution scale s are num_residual_units with
    filter size = filters[s]. strides[s] determine the downsampling factor
    at each resolution scale.

    [1] J. Long et al. Fully convolutional networks for semantic segmentation.
        CVPR 2015.
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
        mode (TYPE, optional): One of the tf.estimator.ModeKeys strings:
            TRAIN, EVAL or PREDICT
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

    conv_params = {'use_bias': use_bias,
                   'kernel_initializer': kernel_initializer,
                   'bias_initializer': bias_initializer,
                   'kernel_regularizer': kernel_regularizer,
                   'bias_regularizer': bias_regularizer}

    x = inputs

    # Inital convolution with filters[0]
    x = tf.layers.conv3d(inputs=x,
                         filters=filters[0],
                         kernel_size=(3, 3, 3),
                         strides=strides[0],
                         padding='same',
                         **conv_params)

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

    # Upscore layers [2] reconstruct the predictions to higher resolution
    # scales
    for res_scale in range(len(filters) - 2, -1, -1):

        with tf.variable_scope('upscore_{}'.format(res_scale)):

            x = upscore_layer_3d(
                inputs=x,
                inputs2=res_scales[res_scale],
                out_filters=num_classes,
                strides=saved_strides[res_scale],
                mode=mode,
                **conv_params)

        tf.logging.info('Decoder at res_scale {} tensor shape: {}'.format(
            res_scale, x.get_shape()))

    # Last convolution
    with tf.variable_scope('last'):
        x = tf.layers.conv3d(inputs=x,
                             filters=num_classes,
                             kernel_size=(1, 1, 1),
                             strides=(1, 1, 1),
                             padding='same',
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
