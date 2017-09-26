from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages

from dltk.core.modules import *


class Upscore(AbstractModule):
    """Upscore module according to J. Long.

    """
    def __init__(self, out_filters, strides, name='upscore'):
        """Constructs an Upscore module

        Parameters
        ----------
        out_filters : int
            number of output filters
        strides : list or tuple
            strides to use for upsampling
        name : string
            name of the module
        """
        self.out_filters = out_filters
        self.strides = strides
        self.in_filters = None
        super(Upscore, self).__init__(name)

    def _build(self, x, x_up, is_training=True):
        """Applies the upscore operation

        Parameters
        ----------
        x : tf.Tensor
            tensor to be upsampled
        x_up : tf.Tensor
            tensor from the same scale to be convolved and added to the upsampled tensor
        is_training : bool
            flag for specifying whether this is training - passed to batch normalization

        Returns
        -------
        tf.Tensor
            output of the upscore operation
        """

        # Compute an up-conv shape dynamically from the input tensor. Input filters are required to be static.
        if self.in_filters is None:
            self.in_filters = x.get_shape().as_list()[-1]
        assert self.in_filters == x.get_shape().as_list()[-1], 'Module was initialised for a different input shape'

        # Account for differences in input and output filters
        if self.in_filters != self.out_filters:
            x = Convolution(self.out_filters, name='up_score_filter_conv')(x)

        t_conv = BilinearUpsample(strides=self.strides)(x)

        conv = Convolution(self.out_filters, 1)(x_up)
        conv = BatchNorm()(conv, is_training)

        return tf.add(t_conv, conv)


class DeepMedic(AbstractModule):
    """FCN module with residual encoder

    This module builds a FCN for segmentation using a residual encoder.
    """
    def __init__(self, num_classes, normal_filters=(30, 30, 40, 40, 40, 40, 50, 50),
                 normal_strides=((1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1)),
                 normal_kernels=((3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)),
                 normal_residuals=(4, 6, 8), normal_input_shape=(25, 25, 25),
                 subsampled_filters=((30, 30, 40, 40, 40, 40, 50, 50),),
                 subsampled_strides=(((1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1)),),
                 subsampled_kernels=(((3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)),),
                 subsampled_residuals=((4, 6, 8),),
                 subsampled_input_shapes=((57, 57, 57),),
                 subsample_factors=((3, 3, 3),),
                 fc_filters=(150, 150),
                 first_fc_kernel=(3, 3, 3),
                 fc_residuals=(2, ),
                 padding='VALID',
                 use_prelu=True,
                 name='deepmedic'):
        """Builds a residual FCN for segmentation

        Parameters
        ----------
        num_classes : int
            number of classes to segment
        num_residual_units : int
            number of residual units per scale
        filters : tuple or list
            number of filters per scale. The first is used for the initial convolution without residual connections
        strides : tuple or list
            strides per scale. The first is used for the initial convolution without residual connections
        relu_leakiness : float
            leakiness of the relus used
        name : string
            name of the network
        """
        self.num_classes = num_classes
        self.normal_filters = normal_filters
        self.normal_strides = normal_strides
        self.normal_kernels = normal_kernels
        self.normal_residuals = normal_residuals
        self.normal_input_shape = normal_input_shape
        self.subsampled_filters = subsampled_filters
        self.subsampled_strides = subsampled_strides
        self.subsampled_kernels = subsampled_kernels
        self.subsampled_residuals = subsampled_residuals
        self.subsample_factors = subsample_factors
        self.subsampled_input_shapes = subsampled_input_shapes
        self.fc_filters = fc_filters
        self.first_fc_kernel = first_fc_kernel
        self.fc_residuals = fc_residuals
        self.padding = padding
        self.use_prelu = use_prelu
        super(DeepMedic, self).__init__(name)

    @classmethod
    def _crop_central_block(cls, inp, size):
        assert all([i >= s for i, s in zip(inp.get_shape().as_list()[1:], size)]), \
            'Output size must no be bigger than input size.'
        slicer = [slice(None)] * len(inp.get_shape().as_list())

        for i in range(len(size)):
            # use i + 1 to account for batch dimension
            start = (inp.get_shape().as_list()[i + 1] - size[i]) // 2
            end = start + size[i]
            slicer[i + 1] = slice(start, end)

        return inp[slicer]

    @classmethod
    def _residual_connection(cls, x, prev_x):
        print(x.get_shape().as_list())
        # crop previous to current size:
        print(prev_x.get_shape().as_list())
        prev_x = cls._crop_central_block(prev_x, x.get_shape().as_list()[1:-1])
        print(prev_x.get_shape().as_list())

        # add prev_x to first channels of x

        to_pad = [[0, 0]] * (len(x.get_shape().as_list()) - 1)
        to_pad += [[0, x.get_shape().as_list()[-1] - prev_x.get_shape().as_list()[-1]]]
        print(to_pad)
        prev_x = tf.pad(prev_x, to_pad)
        print(prev_x.get_shape().as_list())

        return x + prev_x

    def _build_normal_pathway(self, inp, is_training=True):
        with tf.variable_scope('normal_pathway'):
            center_crop = self._crop_central_block(inp, self.normal_input_shape)

            layers = []

            x = center_crop
            for i in range(len(self.normal_filters)):
                with tf.variable_scope('layer_{}'.format(i)):
                    layers.append(x)
                    if i > 0:
                        x = BatchNorm()(x, is_training)
                        x = PReLU()(x) if self.use_prelu else leaky_relu(x, 0.01)
                    x = Convolution(self.normal_filters[i], self.normal_kernels[i], self.normal_strides[i],
                                    padding=self.padding)(x)
                    # TODO: add pooling and dropout?!
                    if i + 1 in self.normal_residuals:
                        x = self._residual_connection(x, layers[i - 1])
        return x

    @classmethod
    def _downsample(cls, x, factor):
        if isinstance(factor, int):
            factor = [factor] * (len(x.get_shape().as_list()) - 2)
        pool_func = tf.nn.avg_pool3d if len(factor) == 3 else tf.nn.avg_pool

        factor = list(factor)

        x = pool_func(x, [1, ] + factor + [1, ], [1, ] + factor + [1, ], 'VALID')
        return x

    @classmethod
    def _upsample(cls, x, factor):
        if isinstance(factor, int):
            factor = [factor] * (len(x.get_shape().as_list()) - 2)

        # TODO: build repeat upsampling

        x = BilinearUpsample(strides=factor)(x)
        return x

    def _build_subsampled_pathways(self, inp, is_training=True):
        pathways = []
        for pathway in range(len(self.subsample_factors)):
            with tf.variable_scope('subsampled_pathway_{}'.format(pathway)):
                center_crop = self._crop_central_block(inp, self.subsampled_input_shapes[pathway])

                layers = []

                x = center_crop
                x = self._downsample(x, self.subsample_factors[pathway])

                for i in range(len(self.subsampled_filters[pathway])):
                    with tf.variable_scope('layer_{}'.format(i)):
                        layers.append(x)
                        if i > 0:
                            x = BatchNorm()(x, is_training)
                            x = PReLU()(x) if self.use_prelu else leaky_relu(x, 0.01)
                        x = Convolution(self.subsampled_filters[pathway][i], self.subsampled_kernels[pathway][i],
                                        self.subsampled_strides[pathway][i], padding=self.padding)(x)
                        # TODO: add pooling and dropout?!
                        if i + 1 in self.subsampled_residuals:
                            x = self._residual_connection(x, layers[i - 1])

                x = self._upsample(x, self.subsample_factors[pathway])
                pathways.append(x)
        return pathways

    def _combine_pathways(self, normal, pathways):
        normal_shape = normal.get_shape().as_list()[1:-1]
        paths = [normal]
        for x in pathways:
            paths.append(self._crop_central_block(x, normal_shape))
        return tf.concat(paths, -1)

    def _build(self, inp, is_training=True):
        """Constructs a ResNetFCN using the input tensor

        Parameters
        ----------
        inp : tf.Tensor
            input tensor
        is_training : bool
            flag to specify whether this is training - passed to batch normalization

        Returns
        -------
        dict
            output dictionary containing:
                - `logits` - logits of the classification
                - `y_prob` - classification probabilities
                - `y_` - prediction of the classification

        """
        outputs = {}

        x = inp

        normal = self._build_normal_pathway(x, is_training)
        pathways = self._build_subsampled_pathways(x, is_training)

        x = self._combine_pathways(normal, pathways)

        layers = []
        for i in range(len(self.fc_filters)):
            with tf.variable_scope('fc_{}'.format(i)):
                layers.append(x)
                if i == 0 and any([k > 1 for k in self.first_fc_kernel]):
                    x_shape = x.get_shape().as_list()
                    # CAUTION: https://docs.python.org/2/faq/programming.html#how-do-i-create-a-multidimensional-list
                    x_pad = [[0, 0] for _ in range(len(x_shape))]
                    for j in range(len(self.first_fc_kernel)):
                        to_pad = (self.first_fc_kernel[j] - 1)
                        x_pad[j + 1][0] = to_pad // 2
                        x_pad[j + 1][1] = to_pad - x_pad[j + 1][0]
                        print(x_pad)
                    x = tf.pad(x, x_pad, mode='SYMMETRIC')

                x = BatchNorm()(x, is_training)
                x = PReLU()(x) if self.use_prelu else leaky_relu(x, 0.01)
                x = Convolution(self.fc_filters[i], self.first_fc_kernel if i == 0 else 1, padding='VALID')(x)
                if i + 1 in self.fc_residuals:
                    x = self._residual_connection(x, layers[i - 1])

        with tf.variable_scope('last'):
            x = BatchNorm()(x, is_training)
            x = PReLU()(x) if self.use_prelu else leaky_relu(x, 0.01)
            x = Convolution(self.num_classes, 1, use_bias=True)(x)

        outputs['logits'] = x
        tf.logging.info('last conv shape %s', x.get_shape())

        with tf.variable_scope('pred'):
            y_prob = tf.nn.softmax(x)
            outputs['y_prob'] = y_prob
            y_ = tf.argmax(x, axis=-1)
            outputs['y_'] = y_

        return outputs