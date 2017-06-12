from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages

from dltk.core.modules.base import AbstractModule
from dltk.core.modules.activations import leaky_relu
from dltk.core.modules.convolution import Convolution
from dltk.core.modules.batch_normalization import BatchNorm


class VanillaResidualUnit(AbstractModule):
    """Vanilla pre-activation residual unit

    pre-activation residual unit as proposed by He, Kaiming, et al. "Identity mappings in deep residual networks."
    ECCV, 2016. - https://link.springer.com/chapter/10.1007/978-3-319-46493-0_38
    """
    def __init__(self, out_filters, kernel_size=3, stride=(1, 1, 1), relu_leakiness=0.01, name='res_unit'):
        """Builds a residual unit

        Parameters
        ----------
        out_filters : int
            number of output filters
        kernel_size : int or tuple or list, optional
            size of the kernel for the convolutions
        stride : int or tuple or list, optional
            stride used for first convolution in unit
        relu_leakiness : float
            leakiness of relu used in unit
        name : string
            name of the module
        """
        if isinstance(kernel_size, int) and isinstance(stride, int):
            kernel_size = np.array([kernel_size] * 3)
            stride = [stride] * 3
        elif isinstance(kernel_size, int):
            kernel_size = np.array([kernel_size] * len(stride))
        elif isinstance(stride, int):
            stride = [stride] * len(kernel_size)

        self.out_filters = out_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.relu_leakiness = relu_leakiness
        self.in_filters = None

        super(VanillaResidualUnit, self).__init__(name=name)

    def _build(self, inp, is_training):
        """Passes a tensor through a residual unit

        Parameters
        ----------
        inp : tf.Tensor
            tensor to be passed through residual unit
        is_training : bool
            flag to toggle training mode - passed to batch normalization

        Returns
        -------
        tf.Tensor
            transformed output of the residual unit
        """
        x = inp
        orig_x = x
        if self.in_filters is None:
            self.in_filters = x.get_shape().as_list()[-1]
        assert(self.in_filters == x.get_shape().as_list()[-1], 'Module was initialised for a different input shape')

        pool_op = tf.nn.max_pool if len(x.get_shape().as_list()) == 4 else tf.nn.max_pool3d

        # Handle strided convolutions
        kernel_size = self.kernel_size
        if np.prod(self.stride) != 1:
            kernel_size = self.stride
            orig_x = pool_op(orig_x, [1, ] + self.stride + [1, ], [1, ] + self.stride + [1, ], 'VALID')

        # Add a convolutional layer
        with tf.variable_scope('sub1'):
            x = BatchNorm()(x, is_training)
            x = leaky_relu(x, self.relu_leakiness)
            x = Convolution(self.out_filters, kernel_size, self.stride)(x)

        # Add a convolutional layer
        with tf.variable_scope('sub2'):
            x = BatchNorm()(x, is_training)
            x = leaky_relu(x, self.relu_leakiness)
            x = Convolution(self.out_filters, self.kernel_size)(x)

        # Add the residual
        with tf.variable_scope('sub_add'):
            # Handle differences in input and output filter sizes
            if self.in_filters < self.out_filters:
                orig_x = tf.pad(orig_x, [[0, 0]] * (len(x.get_shape().as_list()) - 1) +
                                         [[int(np.floor((self.out_filters - self.in_filters) / 2.)),
                                          int(np.ceil((self.out_filters - self.in_filters) / 2.))]])
            elif self.in_filters > self.out_filters:
                orig_x = Convolution(self.out_filters, [1] * len(self.kernel_size), 1)(orig_x)

            x += orig_x
        return x