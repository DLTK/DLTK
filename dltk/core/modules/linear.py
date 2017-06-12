from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf

from dltk.core.modules.base import AbstractModule


class Linear(AbstractModule):
    """Linear layer module

    This module builds a linear layer

    """
    def __init__(self, out_units, use_bias=True, name='linear'):
        """Constructs linear layer

        Parameters
        ----------
        out_units : int
            number of output units
        use_bias : bool, optional
            flag to toggle the addition of a bias
        name : string
            name of the module
        """
        self.out_units = out_units
        self.in_units = None
        self.use_bias = use_bias

        super(Linear, self).__init__(name=name)

    def _build(self, inp):
        """Applies the linear layer operation to an input tensor

        Parameters
        ----------
        inp : tf.Tensor
            input tensor


        Returns
        -------
        tf.Tensor
            transformed tensor

        """
        assert(len(inp.get_shape().as_list())  == 2, 'Layer needs 2D input.')

        self.in_shape = tuple(inp.get_shape().as_list())
        if self.in_units is None:
            self.in_units = self.in_shape[-1]

        assert(self.in_units == self.in_shape[-1], 'Layer was initialised for a different number of input units.')

        w_shape = (self.in_units, self.out_units)

        self._w = tf.get_variable("w", shape=w_shape, initializer=tf.uniform_unit_scaling_initializer(),
                                  collections=self.WEIGHT_COLLECTIONS)
        self.variables.append(self._w)

        if self.use_bias:
            self._b = tf.get_variable("b", shape=(self.out_units,), initializer=tf.constant_initializer(),
                                      collections=self.BIAS_COLLECTIONS)
            self.variables.append(self._b)
            outp = tf.nn.xw_plus_b(inp, self._w, self._b, 'linear')
        else:
            outp = tf.matmul(inp, self._w, 'linear')

        return outp