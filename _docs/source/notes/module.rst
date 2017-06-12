Modules
=======

Modules build the foundation of DLTK and are mostly used to implement layers of neural networks. Modules are implemented
as objects which handle variable sharing. To do this we followed the style of `Sonnet <https://github.com/deepmind/sonnet>`_
and used `tf.make_template <https://www.tensorflow.org/api_docs/python/tf/make_template>`_. To implement a new module
you simply need to inherit from the :class:`~dltk.core.modules.base.AbstractModule` class and overwrite the ``__init__`` and
``_build`` methods.

A simple example is the :class:`~dltk.core.modules.linear.Linear` layer that implements a simple matrix multiplication.
The ``__init__`` method stores all necessary parameters for the module like the number of output units and whether to
add a bias or not. It then calls the super class's ``__init__`` function to build the template:
::
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


The ``_build`` function then handles the actual implementation of the module. All variables have to be created with
``tf.get_variable``. Variable sharing and scoping is automatically handled by ``tf.make_template`` which is called in
the :class:`~dltk.core.modules.base.AbstractModule` class.
::
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

The output of this function is passed through wrappers and returned when an instance of this module is called
::
  linear = Linear(100)
  outp = linear(x) # <-- this is the same outp as returned in _build