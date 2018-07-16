from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
import numpy as np


def vanilla_residual_unit_3d(inputs,
                             out_filters,
                             kernel_size=(3, 3, 3),
                             strides=(1, 1, 1),
                             mode=tf.estimator.ModeKeys.EVAL,
                             use_bias=False,
                             activation=tf.nn.relu6,
                             kernel_initializer=tf.initializers.variance_scaling(distribution='uniform'),
                             bias_initializer=tf.zeros_initializer(),
                             kernel_regularizer=None,
                             bias_regularizer=None):
    """Implementation of a 3D residual unit according to [1]. This
        implementation supports strided convolutions and automatically
        handles different input and output filters.

    [1] K. He et al. Identity Mappings in Deep Residual Networks. ECCV 2016.

    Args:
        inputs (tf.Tensor): Input tensor to the residual unit. Is required to
            have a rank of 5 (i.e. [batch, x, y, z, channels]).
        out_filters (int): Number of convolutional filters used in
            the sub units.
        kernel_size (tuple, optional): Size of the convoltional kernels
            used in the sub units
        strides (tuple, optional): Convolution strides in (x,y,z) of sub
            unit 0. Allows downsampling of the input tensor via strides
            convolutions.
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

    pool_op = tf.layers.max_pooling3d

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
    orig_x = x

    # Handle strided convolutions
    if np.prod(strides) != 1:
        orig_x = pool_op(inputs=orig_x,
                         pool_size=strides,
                         strides=strides,
                         padding='valid')

    # Sub unit 0
    with tf.variable_scope('sub_unit0'):

        # Adjust the strided conv kernel size to prevent losing information
        k = [s * 2 if s > 1 else k for k, s in zip(kernel_size, strides)]

        x = tf.layers.batch_normalization(
            x, training=mode == tf.estimator.ModeKeys.TRAIN)
        x = activation(x)

        x = tf.layers.conv3d(
            inputs=x,
            filters=out_filters,
            kernel_size=k, strides=strides,
            **conv_params)

    # Sub unit 1
    with tf.variable_scope('sub_unit1'):
        x = tf.layers.batch_normalization(
            x, training=mode == tf.estimator.ModeKeys.TRAIN)
        x = activation(x)

        x = tf.layers.conv3d(
            inputs=x,
            filters=out_filters,
            kernel_size=kernel_size,
            strides=(1, 1, 1),
            **conv_params)

    # Add the residual
    with tf.variable_scope('sub_unit_add'):

        # Handle differences in input and output filter sizes
        if in_filters < out_filters:
            orig_x = tf.pad(
                tensor=orig_x,
                paddings=[[0, 0]] * (len(x.get_shape().as_list()) - 1) + [[
                    int(np.floor((out_filters - in_filters) / 2.)),
                    int(np.ceil((out_filters - in_filters) / 2.))]])

        elif in_filters > out_filters:
            orig_x = tf.layers.conv3d(
                inputs=orig_x,
                filters=out_filters,
                kernel_size=kernel_size,
                strides=(1, 1, 1),
                **conv_params)
        x += orig_x

    return x


def residual_unit_3d(inputs,
                     out_filters,
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

    [1] K. He et al. Identity Mappings in Deep Residual Networks. ECCV 2016.

    Args:
        inputs (tf.Tensor): Input tensor to the residual unit. Is required to
            have a rank of 5 (i.e. [batch, x, y, z, channels]).
        out_filters (int): Number of convolutional filters used in
            the sub units.
        num_convolutions (int): Number of convolutions to skip over.
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
    orig_x = x

    # Sub feature extraction units
    for i in range(num_convolutions):
        with tf.variable_scope('sub_unit_{}'.format(i)):

            x = tf.layers.batch_normalization(
                x, training=mode == tf.estimator.ModeKeys.TRAIN)
            
            x = activation(x)

            x = tf.layers.conv3d(
                inputs=x,
                filters=out_filters,
                kernel_size=kernel_size,
                **conv_params)

    # Add the residual
    with tf.variable_scope('sub_unit_add'):
        # Handle differences in input and output filter sizes
        if in_filters < out_filters:
            orig_x = tf.pad(
                tensor=orig_x,
                paddings=[[0, 0]] * (len(x.get_shape().as_list()) - 1) + [[
                    int(np.floor((out_filters - in_filters) / 2.)),
                    int(np.ceil((out_filters - in_filters) / 2.))]])

        elif in_filters > out_filters:
            orig_x = tf.layers.conv3d(
                inputs=orig_x,
                filters=out_filters,
                kernel_size=kernel_size,
                **conv_params)
        x += orig_x

    return x


def se_residual_unit_3d(inputs,
                        out_filters,
                        num_convolutions=2,
                        kernel_size=(3, 3, 3),
                        reduction_ratio=16,
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

    [1] K. He et al. Identity Mappings in Deep Residual Networks. ECCV 2016.
    [2] J. Hu et al. Squeeze-and-excitation networks. CVPR 2017.

    Args:
        inputs (tf.Tensor): Input tensor to the residual unit. Is required to
            have a rank of 5 (i.e. [batch, x, y, z, channels]).
        out_filters (int): Number of convolutional filters used in
            the sub units.
        num_convolutions (int): Number of convolutions to skip over.
        kernel_size (tuple, optional): Size of the convoltional kernels
            used in the sub units
        reduction_ratio (int): Ratio to reduce filters in SE-MLP with.
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
    orig_x = x

    # Sub feature extraction units
    for i in range(num_convolutions):
        with tf.variable_scope('sub_unit_{}'.format()):

            x = tf.layers.batch_normalization(
                x, training=mode == tf.estimator.ModeKeys.TRAIN)
            x = activation(x)

            x = tf.layers.conv3d(
                inputs=x,
                filters=out_filters,
                kernel_size=kernel_size,
                **conv_params)

    with tf.variable_scope('sub_se_unit'):
        pooled = tf.reduce_mean(x, axis=[1, 2, 3])

        scale = tf.layers.dense(pooled, out_filters // reduction_ratio)

        scale = tf.nn.relu(scale)

        scale = tf.layers.dense(scale, out_filters)

        scale = tf.nn.sigmoid(scale)

        scale = tf.reshape(scale, [-1, 1, 1, 1, out_filters])

        x = scale * x

    # Add the residual
    with tf.variable_scope('sub_unit_add'):
        # Handle differences in input and output filter sizes
        if in_filters < out_filters:
            orig_x = tf.pad(
                tensor=orig_x,
                paddings=[[0, 0]] * (len(x.get_shape().as_list()) - 1) + [[
                    int(np.floor((out_filters - in_filters) / 2.)),
                    int(np.ceil((out_filters - in_filters) / 2.))]])

        elif in_filters > out_filters:
            orig_x = tf.layers.conv3d(
                inputs=orig_x,
                filters=out_filters,
                kernel_size=kernel_size,
                strides=(1, 1, 1),
                **conv_params)
        x += orig_x

    return x


def resnext_unit_3d(inputs,
                    out_filters,
                    cardinality=32,
                    bottleneck_channels=4,
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

    [1] S. Xie et al. Aggregated Residual Transformations for Deep Neural
        Networks. arXiv 2016.

    Args:
        inputs (tf.Tensor): Input tensor to the residual unit. Is required to
            have a rank of 5 (i.e. [batch, x, y, z, channels]).
        out_filters (int): Number of convolutional filters used in
            the sub units.
        cardinality (int): Number of convolution groups in the resnext unit.
        bottleneck_channels (int): Number of channels per group.
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
    orig_x = x

    # if TF group convolutions exist use:
    # bottleneck_size = bottleneck_channels * cardinality

    with tf.variable_scope('sub_unit_convs'):
        x = tf.layers.batch_normalization(
            x, training=mode == tf.estimator.ModeKeys.TRAIN)
        x = activation(x)

        pre_split = x

        groups = []
        for group in range(cardinality):
            with tf.variable_scope('group_{}'.format(group)):
                x = tf.layers.conv3d(
                    inputs=pre_split,
                    filters=bottleneck_channels,
                    kernel_size=1,
                    **conv_params)

                x = tf.layers.conv3d(
                    inputs=x,
                    filters=bottleneck_channels,
                    kernel_size=kernel_size,
                    **conv_params)

                groups += [x]

        x = tf.concat(groups, 5)

        x = tf.layers.conv3d(
            inputs=x,
            filters=out_filters,
            kernel_size=1,
            **conv_params)

    # Add the residual
    with tf.variable_scope('sub_unit_add'):
        # Handle differences in input and output filter sizes
        if in_filters < out_filters:
            orig_x = tf.pad(
                tensor=orig_x,
                paddings=[[0, 0]] * (len(x.get_shape().as_list()) - 1) + [[
                    int(np.floor((out_filters - in_filters) / 2.)),
                    int(np.ceil((out_filters - in_filters) / 2.))]])

        elif in_filters > out_filters:
            orig_x = tf.layers.conv3d(
                inputs=orig_x,
                filters=out_filters,
                kernel_size=kernel_size,
                strides=(1, 1, 1),
                **conv_params)
        x += orig_x

    return x


def se_resnext_unit_3d(inputs,
                       out_filters,
                       cardinality=32,
                       bottleneck_channels=4,
                       reduction_ratio=16,
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

    [1] S. Xie et al. Aggregated Residual Transformations for Deep Neural
        Networks. arXiv 2016.
    [2] J. Hu et al. Squeeze-and-excitation networks. CVPR 2017.

    Args:
        inputs (tf.Tensor): Input tensor to the residual unit. Is required to
            have a rank of 5 (i.e. [batch, x, y, z, channels]).
        out_filters (int): Number of convolutional filters used in
            the sub units.
        cardinality (int): Number of convolution groups in the resnext unit.
        bottleneck_channels (int): Number of channels per group.
        reduction_ratio (int): Ratio to reduce filters in SE-MLP with.
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
    orig_x = x

    # if TF group convolutions exist use:
    # bottleneck_size = bottleneck_channels * cardinality

    with tf.variable_scope('sub_unit_convs'):
        x = tf.layers.batch_normalization(
            x, training=mode == tf.estimator.ModeKeys.TRAIN)
        x = activation(x)

        pre_split = x

        groups = []
        for group in range(cardinality):
            with tf.variable_scope('group_{}'.format(group)):
                x = tf.layers.conv3d(
                    inputs=pre_split,
                    filters=bottleneck_channels,
                    kernel_size=1,
                    **conv_params)

                x = tf.layers.conv3d(
                    inputs=x,
                    filters=bottleneck_channels,
                    kernel_size=kernel_size,
                    **conv_params)

                groups += [x]

        x = tf.concat(groups, 5)

        x = tf.layers.conv3d(
            inputs=x,
            filters=out_filters,
            kernel_size=1,
            **conv_params)

    with tf.variable_scope('sub_se_unit'):
        pooled = tf.reduce_mean(x, axis=[1, 2, 3])

        scale = tf.layers.dense(pooled, out_filters // reduction_ratio)

        scale = tf.nn.relu(scale)

        scale = tf.layers.dense(scale, out_filters)

        scale = tf.nn.sigmoid(scale)

        scale = tf.reshape(scale, [-1, 1, 1, 1, out_filters])

        x = scale * x

    # Add the residual
    with tf.variable_scope('sub_unit_add'):
        # Handle differences in input and output filter sizes
        if in_filters < out_filters:
            orig_x = tf.pad(
                tensor=orig_x,
                paddings=[[0, 0]] * (len(x.get_shape().as_list()) - 1) + [[
                    int(np.floor((out_filters - in_filters) / 2.)),
                    int(np.ceil((out_filters - in_filters) / 2.))]])

        elif in_filters > out_filters:
            orig_x = tf.layers.conv3d(
                inputs=orig_x,
                filters=out_filters,
                kernel_size=kernel_size,
                strides=(1, 1, 1),
                **conv_params)
        x += orig_x

    return x
