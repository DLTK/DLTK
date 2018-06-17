# WARNING/NOTE
# This implementation is work in progress and an attempt to implement a
# scalable version of the original DeepMedic [1] source. It will NOT
# yield the same accuracy performance as described in the paper.
# If you are running comparative experiments, please refer to the
# original code base in [1].
#
# [1] https://github.com/Kamnitsask/deepmedic

from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf

from dltk.core.upsample import linear_upsample_3d
from dltk.core.activations import prelu, leaky_relu


def crop_central_block(x, size):
    assert all([i >= s for i, s in zip(x.get_shape().as_list()[1:], size)]), \
        'Output size must not be bigger than input size. But was {} compared ' \
        'to {}'.format(x.get_shape().as_list()[1:], size)

    slicer = [slice(None)] * len(x.get_shape().as_list())

    for i in range(len(size)):
        # use i + 1 to account for batch dimension
        start = (x.get_shape().as_list()[i + 1] - size[i]) // 2
        end = start + size[i]
        slicer[i + 1] = slice(start, end)

    return x[slicer]


def deepmedic_3d(inputs, num_classes,
                 normal_filters=(30, 30, 40, 40, 40, 40, 50, 50),
                 normal_strides=((1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1),
                                 (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1)),
                 normal_kernels=((3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3),
                                 (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)),
                 normal_residuals=(4, 6, 8),
                 normal_input_shape=(25, 25, 25),
                 subsampled_filters=((30, 30, 40, 40, 40, 40, 50, 50),),
                 subsampled_strides=(((1, 1, 1), (1, 1, 1), (1, 1, 1),
                                      (1, 1, 1), (1, 1, 1), (1, 1, 1),
                                      (1, 1, 1), (1, 1, 1)),),
                 subsampled_kernels=(((3, 3, 3), (3, 3, 3), (3, 3, 3),
                                      (3, 3, 3), (3, 3, 3), (3, 3, 3),
                                      (3, 3, 3), (3, 3, 3)),),
                 subsampled_residuals=((4, 6, 8),),
                 subsampled_input_shapes=((57, 57, 57),),
                 subsample_factors=((3, 3, 3),),
                 fc_filters=(150, 150),
                 first_fc_kernel=(3, 3, 3),
                 fc_residuals=(2, ),
                 padding='VALID',
                 use_prelu=True,
                 mode=tf.estimator.ModeKeys.EVAL,
                 use_bias=True,
                 kernel_initializer=tf.initializers.variance_scaling(distribution='uniform'),
                 bias_initializer=tf.zeros_initializer(),
                 kernel_regularizer=None,
                 bias_regularizer=None):
    """
    Image segmentation network based on a DeepMedic architecture [1, 2].
    Downsampling of features is done via strided convolutions. The architecture
    uses multiple processing paths with different resolutions. The different
    pathways are concatenated and then fed to the convolutional fc layers.

    [1] Konstantinos Kamnitsas et al. Efficient Multi-Scale 3D CNN with Fully
        Connected CRF for Accurate Brain Lesion Segmentation. Medical Image
        Analysis, 2016.
    [2] Konstantinos Kamnitsas et al. Multi-Scale 3D CNNs for segmentation of
        brain Lesions in multi-modal MRI. ISLES challenge, MICCAI 2015.

    Note: We are currently using bilinear upsampling whereas the original
    implementation (https://github.com/Kamnitsask/deepmedic) uses repeat
    upsampling.

    Args:
        inputs (tf.Tensor): Input feature tensor to the network (rank 5
            required).
        num_classes (int): Number of output classes.
        normal_filters (array_like, optional): Number of filters for each layer
            for normal path.
        normal_strides (array_like, optional): Strides for each layer for
            normal path.
        normal_kernels (array_like, optional): Kernel size for each layer for
            normal path.
        normal_residuals (array_like, optional): Location of residual
            connections for normal path.
        normal_input_shape (array_like, optional): Shape of input to normal
            path. Input to the network is center cropped to this shape.
        subsampled_filters (array_like, optional): Number of filters for each
            layer for each subsampled path.
        subsampled_strides (array_like, optional): Strides for each layer for
            each subsampled path.
        subsampled_kernels (array_like, optional): Kernel size for each layer
            for each subsampled path.
        subsampled_residuals (array_like, optional): Location of residual
            connections for each subsampled path.
        subsampled_input_shapes (array_like, optional): Shape of input to
            subsampled paths. Input to the network is downsampled and then
            center cropped to this shape.
        subsample_factors (array_like, optional): Downsampling factors for
            each subsampled path.
        fc_filters (array_like, optional): Number of filters for the fc layers.
        first_fc_kernel (array_like, optional): Shape of the kernel of the
            first fc layer.
        fc_residuals (array_like, optional): Location of residual connections
            for the fc layers.
        padding (string, optional): Type of padding used for convolutions.
            Standard is `VALID`
        use_prelu (bool, optional): Flag to enable PReLU activation.
            Alternatively leaky ReLU is used. Defaults to `True`.
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
        dict: dictionary of output tensors

    """
    outputs = {}
    assert len(normal_filters) == len(normal_strides)
    assert len(normal_filters) == len(normal_kernels)
    assert len(inputs.get_shape().as_list()) == 5, \
        'inputs are required to have a rank of 5.'

    conv_params = {'use_bias': use_bias,
                   'kernel_initializer': kernel_initializer,
                   'bias_initializer': bias_initializer,
                   'kernel_regularizer': kernel_regularizer,
                   'bias_regularizer': bias_regularizer,
                   'padding': padding}

    def _residual_connection(x, prev_x):
        # crop previous to current size:
        prev_x = crop_central_block(prev_x, x.get_shape().as_list()[1:-1])

        # add prev_x to first channels of x

        to_pad = [[0, 0]] * (len(x.get_shape().as_list()) - 1)
        to_pad += [[0, x.get_shape().as_list()[-1] -
                    prev_x.get_shape().as_list()[-1]]]
        prev_x = tf.pad(prev_x, to_pad)

        return x + prev_x

    def _build_normal_pathway(x):
        with tf.variable_scope('normal_pathway'):
            tf.logging.info('Building normal pathway')
            center_crop = crop_central_block(x, normal_input_shape)
            tf.logging.info('Input is {}'.format(
                center_crop.get_shape().as_list()))

            layers = []

            x = center_crop
            for i in range(len(normal_filters)):
                with tf.variable_scope('layer_{}'.format(i)):
                    layers.append(x)
                    if i > 0:
                        x = tf.layers.batch_normalization(
                            x, training=mode == tf.estimator.ModeKeys.TRAIN)
                        x = prelu(x) if use_prelu else leaky_relu(x, 0.01)
                    x = tf.layers.conv3d(x,
                                         normal_filters[i],
                                         normal_kernels[i],
                                         normal_strides[i],
                                         **conv_params)
                    # TODO: add pooling and dropout?!
                    if i + 1 in normal_residuals:
                        x = _residual_connection(x, layers[i - 1])
                    tf.logging.info('Output of layer {} is {}'.format(
                        i, x.get_shape().as_list()))
        tf.logging.info('Output is {}'.format(x.get_shape().as_list()))
        return x

    def _downsample(x, factor):
        if isinstance(factor, int):
            factor = [factor] * (len(x.get_shape().as_list()) - 2)
        pool_func = tf.nn.avg_pool3d

        factor = list(factor)

        x = pool_func(x, [1, ] + factor + [1, ], [1, ] + factor + [1, ],
                      'VALID')
        return x

    def _build_subsampled_pathways(x):
        pathways = []
        for pathway in range(len(subsample_factors)):
            with tf.variable_scope('subsampled_pathway_{}'.format(pathway)):
                tf.logging.info(
                    'Building subsampled pathway {}'.format(pathway))
                center_crop = crop_central_block(
                    x, subsampled_input_shapes[pathway])
                tf.logging.info('Input is {}'.format(
                    center_crop.get_shape().as_list()))

                layers = []

                x = center_crop
                x = _downsample(x, subsample_factors[pathway])
                tf.logging.info('Downsampled input is {}'.format(
                    x.get_shape().as_list()))

                for i in range(len(subsampled_filters[pathway])):
                    with tf.variable_scope('layer_{}'.format(i)):
                        layers.append(x)
                        if i > 0:
                            x = tf.layers.batch_normalization(
                                x, training=mode == tf.estimator.ModeKeys.TRAIN)
                            x = prelu(x) if use_prelu else leaky_relu(x, 0.01)
                        x = tf.layers.conv3d(x, subsampled_filters[pathway][i],
                                             subsampled_kernels[pathway][i],
                                             subsampled_strides[pathway][i],
                                             **conv_params)
                        # TODO: add pooling and dropout?!
                        if i + 1 in subsampled_residuals:
                            x = _residual_connection(x, layers[i - 1])
                        tf.logging.info('Output of layer {} is {}'.format(
                            i, x.get_shape().as_list()))

                x = _upsample(x, subsample_factors[pathway])
                tf.logging.info('Output is {}'.format(x.get_shape().as_list()))
                pathways.append(x)
        return pathways

    def _upsample(x, factor):
        if isinstance(factor, int):
            factor = [factor] * (len(x.get_shape().as_list()) - 2)

        # TODO: build repeat upsampling

        x = linear_upsample_3d(x, strides=factor)
        return x

    x = inputs

    normal = _build_normal_pathway(x)
    pathways = _build_subsampled_pathways(x)

    normal_shape = normal.get_shape().as_list()[1:-1]
    paths = [normal]
    for x in pathways:
        paths.append(crop_central_block(x, normal_shape))

    x = tf.concat(paths, -1)

    layers = []
    for i in range(len(fc_filters)):
        with tf.variable_scope('fc_{}'.format(i)):
            layers.append(x)
            if i == 0 and any([k > 1 for k in first_fc_kernel]):
                x_shape = x.get_shape().as_list()
                # CAUTION: https://docs.python.org/2/faq/programming.html#how-do-i-create-a-multidimensional-list
                x_pad = [[0, 0] for _ in range(len(x_shape))]
                for j in range(len(first_fc_kernel)):
                    to_pad = (first_fc_kernel[j] - 1)
                    x_pad[j + 1][0] = to_pad // 2
                    x_pad[j + 1][1] = to_pad - x_pad[j + 1][0]
                    print(x_pad)
                x = tf.pad(x, x_pad, mode='SYMMETRIC')

            x = tf.layers.batch_normalization(
                x, training=mode == tf.estimator.ModeKeys.TRAIN)
            x = prelu(x) if use_prelu else leaky_relu(x, 0.01)
            x = tf.layers.conv3d(x, fc_filters[i],
                                 first_fc_kernel if i == 0 else 1,
                                 **conv_params)
            if i + 1 in fc_residuals:
                x = _residual_connection(x, layers[i - 1])

    with tf.variable_scope('last'):
        x = tf.layers.batch_normalization(
            x, training=mode == tf.estimator.ModeKeys.TRAIN)
        x = prelu(x) if use_prelu else leaky_relu(x, 0.01)
        conv_params['use_bias'] = True
        x = tf.layers.conv3d(x, num_classes, 1, **conv_params)

    outputs['logits'] = x
    tf.logging.info('last conv shape %s', x.get_shape())

    with tf.variable_scope('pred'):
        y_prob = tf.nn.softmax(x)
        outputs['y_prob'] = y_prob
        y_ = tf.argmax(x, axis=-1)
        outputs['y_'] = y_

    return outputs
