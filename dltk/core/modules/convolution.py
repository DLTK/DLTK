from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import numpy as np

from dltk.core.modules.base import AbstractModule


class Convolution(AbstractModule):
    """Convolution module

    This module builds a n-D convolution based on the dimensionality of the input and applies it to the input.

    """
    def __init__(self, out_filters, filter_shape=3, strides=1, dilation_rate=1, padding='SAME', use_bias=False,
                 name='conv'):
        """Constructs the convolution template

        Parameters
        ----------
        out_filters : int
            number of output filters
        filter_shape : int or tuple or list, optional
            shape of the filter to use for the convolution
        strides : int or tuple or list, optional
            stride of the convolution operation
        dilation_rate : tuple or list, optional
            dilation rate used for dilated convolution. If used, stride must be 1
        padding : str
            edge padding of convolutions, one of 'SAME' or 'VALID'
        use_bias : bool
            flag to toggle addition of a bias per output filter
        name : string
            name of the module
        """

        if isinstance(filter_shape, int) and isinstance(strides, int) and isinstance(dilation_rate, int):
            filter_shape = np.array([filter_shape] * 3)
            strides = [strides] * 3
            dilation_rate = [dilation_rate] * 3
        elif isinstance(filter_shape, int) and isinstance(strides, (list, tuple)) and isinstance(dilation_rate, int):
            filter_shape = np.array([filter_shape] * len(strides))
            dilation_rate = [dilation_rate] * len(strides)
        elif isinstance(filter_shape, int) and isinstance(dilation_rate, (list, tuple)) and isinstance(strides, int):
            filter_shape = np.array([filter_shape] * len(dilation_rate))
            strides = [strides] * len(dilation_rate)
        elif (isinstance(filter_shape, (list, tuple, np.ndarray))
              and isinstance(dilation_rate, int) and isinstance(strides, int)):
            dilation_rate = [dilation_rate] * len(filter_shape)
            strides = [strides] * len(filter_shape)
        elif isinstance(strides, int):
            strides = [strides] * len(filter_shape)
        elif isinstance(dilation_rate, int):
            dilation_rate = [dilation_rate] * len(filter_shape)
        else:
            raise Exception('Could not infer the dimensionality of the operation or both strides and dilation was'
                            'passed as list or tuple')

        assert(len(strides) == len(filter_shape), 'Stride len must match len of filter shape')
        assert(len(strides) == len(dilation_rate), 'Dilation rate and stride len must match')
        assert(np.prod(dilation_rate) == 1 or np.prod(strides) == 1, 'Dilation rate or strides must be 1')
        assert(padding == 'SAME' or padding == 'VALID', 'Padding must be either SAME or VALID')

        self.filter_shape = filter_shape
        self.in_shape = None
        self.in_filters = None
        self.out_filters = out_filters
        self.strides = strides
        self.use_bias = use_bias
        self.dilation_rate = dilation_rate
        self.padding = padding

        self._rank = len(list(self.filter_shape))
        assert (self._rank < 4, 'Convolutions are only supported up to 3D')

        super(Convolution, self).__init__(name=name)

    def _build(self, inp):
        """Applies a convolution operation to an input tensor

        Parameters
        ----------
        inp : tf.Tensor
            input tensor to be convolved

        Returns
        -------
        tf.Tensor
            convolved tensor

        """
        assert((len(inp.get_shape().as_list()) - 2)  == self._rank,
               'The input has {} dimensions but this is a {}D convolution'.format(len(inp.get_shape().as_list()),
                                                                                  self._rank))

        self.in_shape = tuple(inp.get_shape().as_list())
        if self.in_filters is None:
            self.in_filters = self.in_shape[-1]
        assert(self.in_filters == self.in_shape[-1], 'Convolution was built for different number of channels')

        self.in_filters = self.in_shape[-1]

        kernel_shape = tuple(list(self.filter_shape) + [self.in_filters, self.out_filters])

        self._k = tf.get_variable("k", shape=kernel_shape, initializer=tf.uniform_unit_scaling_initializer(),
                                  collections=self.WEIGHT_COLLECTIONS)
        self.variables.append(self._k)
        outp = tf.nn.convolution(inp, self._k, padding=self.padding, strides=self.strides,
                                 dilation_rate=self.dilation_rate, name='conv')

        if self.use_bias:
            self._b = tf.get_variable("b", shape=(self.out_filters,), initializer=tf.constant_initializer(),
                                      collections=self.BIAS_COLLECTIONS)
            self.variables.append(self._b)
            outp += self._b

        return outp