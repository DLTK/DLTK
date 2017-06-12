from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf

from dltk.core.modules.base import AbstractModule


class TransposedConvolution(AbstractModule):
    """Tranposed convolution module

    This build a 2D or 3D transposed convolution based on the dimensionality of the input

    """
    def __init__(self, out_filters, strides=(1, 1, 1), filter_shape=None, use_bias=False, name='conv_transposed'):
        """Constructs a transposed convolution

        The kernel shape is defined as 2 * stride for stride > 1

        Parameters
        ----------
        out_filters : int
            number of output filters
        strides : tuple or list, optional
            strides used for the transposed convolution
        use_bias : bool
            flag to toggle whether a bias is added to the output
        name : string
            name of the module
        """
        self.in_shape = None
        self.in_filters = None
        self.out_filters = out_filters
        self.out_shape = None
        self.strides = strides
        self.use_bias = use_bias
        self.filter_shape = filter_shape
        self.full_strides =[1,] + list(self.strides) + [1,]

        self._rank = len(list(self.strides))
        assert(1 < self._rank < 4, 'Transposed convolutions are only supported in 2D and 3D')

        super(TransposedConvolution, self).__init__(name=name)

    def _get_kernel(self):
        """Builds the kernel for the transposed convolution

        Returns
        -------
        tf.Variable
            kernel for the transposed convolution

        """
        kernel_shape = tuple(self.up_spatial_shape + [self.out_filters, self.in_filters])

        k = tf.get_variable("k", shape=kernel_shape, initializer=tf.uniform_unit_scaling_initializer(),
                            collections=self.WEIGHT_COLLECTIONS)

        return k

    def _build(self, inp):
        """Applies a transposed convolution to the input tensor

        Parameters
        ----------
        inp : tf.Tensor
            input tensor

        Returns
        -------
        tf.Tensor
            output of transposed convolution

        """
        assert((len(inp.get_shape().as_list()) - 2)  == self._rank,
               'The input has {} dimensions but this is a {}D convolution'.format(len(inp.get_shape().as_list()),
                                                                                  self._rank))

        self.in_shape = tuple(inp.get_shape().as_list())
        if self.in_filters is None:
            self.in_filters = self.in_shape[-1]
        assert (self.in_filters == self.in_shape[-1], 'Convolution was built for different number of channels')

        inp_shape = tf.shape(inp)

        if self.filter_shape is None:
            self.up_spatial_shape = [2 * s if s > 1 else 1 for s in self.strides]
        else:
            self.up_spatial_shape = self.filter_shape

        self.out_shape = [inp_shape[i] * self.full_strides[i] for i in range(len(self.in_shape) - 1)] + [self.out_filters,]



        self._k = self._get_kernel()

        self.variables.append(self._k)

        conv_op = tf.nn.conv3d_transpose
        if self._rank == 2:
            conv_op = tf.nn.conv2d_transpose

        outp = conv_op(inp, self._k, output_shape=self.out_shape, strides=self.full_strides, padding='SAME',
                       name='conv_tranposed')

        if self.use_bias:
            self._b = tf.get_variable("b", shape=(self.out_filters,), initializer=tf.constant_initializer())
            self.variables.append(self._b)
            outp += self._b
        outp.set_shape([self.in_shape[i] * self.full_strides[i] if isinstance(self.in_shape[i], int) else None
                        for i in range(len(self.in_shape) - 1)] + [self.out_filters,])

        return outp