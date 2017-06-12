from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import numpy as np

from dltk.core.modules.tranposed_convolution import TransposedConvolution


class BilinearUpsample(TransposedConvolution):
    """Bilinear upsampling module

    This module builds a bilinear upsampling filter and uses it to upsample the input tensor.

    """
    def __init__(self, trainable=False, strides=(2, 2, 2), use_bias=False, name='bilinear_upsampling'):
        """Constructs the bilinear upsampling module

        Parameters
        ----------
        trainable : bool, optional
            flag to toggle whether the filter is trainable
        strides : tuple or list, optional
            strides to use for upsampling, also specify the upsampling factor
        use_bias : bool, optional
            flag to toggle the addition of a bias to the output
        name : string, optional
            name for this module
        """
        self.trainable = trainable
        super(BilinearUpsample, self).__init__(None, strides=strides, use_bias=use_bias, name=name)

    def _get_kernel(self):
        """builds kernel for bilinear upsampling"""
        kernel_shape = tuple(self.up_spatial_shape + [self.out_filters, self.in_filters])
        size = self.up_spatial_shape
        factor = (np.array(size) + 1) // 2
        center = np.zeros_like(factor, np.float)

        for i in range(len(factor)):
            if size[i] % 2 == 1:
                center[i] = factor[i] - 1
            else:
                center[i] = factor[i] - 0.5

        weights = np.zeros(kernel_shape)
        if self._rank == 2:
            og = np.ogrid[:size[0], :size[1]]
            x_filt = (1 - abs(og[0] - center[0]) / np.float(factor[0]))
            y_filt = (1 - abs(og[1] - center[1]) / np.float(factor[1]))

            filt = x_filt * y_filt

            for i in range(self.out_filters):
                weights[:, :, i, i] = filt
        else:
            og = np.ogrid[:size[0], :size[1], :size[2]]
            x_filt = (1 - abs(og[0] - center[0]) / np.float(factor[0]))
            y_filt = (1 - abs(og[1] - center[1]) / np.float(factor[1]))
            z_filt = (1 - abs(og[2] - center[2]) / np.float(factor[2]))

            filt = x_filt * y_filt * z_filt

            for i in range(self.out_filters):
                weights[:, :, :, i, i] = filt

        init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)
        return tf.get_variable(name="upsampling_filter", initializer=init, shape=weights.shape,
                               trainable=self.trainable,
                               collections=self.WEIGHT_COLLECTIONS if self.trainable else self.MODEL_COLLECTIONS)

    def _build(self, inp):
        """Applies bilinear upsampling to an input tensor

        Parameters
        ----------
        inp : tf.Tensor
            input to upsample

        Returns
        -------
        tf.Tensor
            upsampled tensor
        """
        self.out_filters = tuple(inp.get_shape().as_list())[-1]
        return super(BilinearUpsample, self)._build(inp)