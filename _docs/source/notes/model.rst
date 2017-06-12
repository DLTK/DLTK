Models
======

Neural network architectures are called models in DLTK. The can be very specific like
:class:`~dltk.models.classification.lenet.LeNet5` or modular like :class:`~dltk.models.classification.resnet.ResNet`.
Each model is implemented identically to modules. However, modules conventially return ``tf.Tensor``objects whereas
models return ``dict`` objects which hold the different outputs of a network. Because of this the structure of model
implementations is again divided into the ``__init__`` and ``_build`` function. For
:class:`~dltk.models.classification.lenet.LeNet5` the init function is very simple:
::
  class LeNet5(AbstractModule):
    """ LeNet5 classification network according to """
    def __init__(self, num_classes=10, name='lenet5'):
        """ Builds the network

        Parameters
        ----------
        num_classes : int
            number of classes to segment
        name : string
            name of the network
        """
        self.num_classes = num_classes
        self.filters = [16,32,100]
        self._rank = None
        super(LeNet5, self).__init__(name)

and the ``_build`` function simply chains multiple modules together and then returns a dict with multiple outputs.
::
      def _build(self, inp, is_training=True):
        """ Constructs a LeNet using the input tensor

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
        if self._rank is None:
            self._rank = len(inp.get_shape().as_list()) - 2
        assert(self._rank == len(inp.get_shape().as_list()) - 2, 'Net was built for a different input size')

        outputs = {}
        pool_op = tf.nn.max_pool if len(inp.get_shape().as_list()) == 4 else tf.nn.max_pool3d

        # MNIST inputs are [batchsize, 28, 28, 1]
        x = inp

        # First conv/pool feature Layer
        x = Convolution(out_filters=self.filters[0],
                        filter_shape=[5] * self._rank,
                        strides=1,
                        padding='VALID',
                        use_bias=True)(x)
        x = tf.nn.tanh(x)

        # When pooling use a kernel size of the size of the strides to not lose information
        x = pool_op(x,
                    ksize=[1] + [2] * self._rank + [1],
                    strides=[1] + [2] * self._rank + [1],
                    padding='VALID')

        # Second conv/pool feature Layer
        x = Convolution(out_filters=self.filters[1],
                        filter_shape=[5] * self._rank,
                        strides=1,
                        padding='VALID',
                        use_bias=True)(x)
        x = tf.nn.tanh(x)
        x = pool_op(x,
                    ksize=[1] + [2] * self._rank + [1],
                    strides=[1] + [2] * self._rank + [1],
                    padding='VALID')

        print(x.get_shape().as_list())
        # First fully connected layer
        x = tf.reshape(x, [tf.shape(x)[0], np.prod(x.get_shape().as_list()[1:])])
        print(x.get_shape().as_list())

        x = tf.layers.dense(inputs=x, units=self.filters[2], activation=tf.nn.tanh)

        # Second fully connected layer, reducing to num_classes
        x = tf.layers.dense(inputs=x, units=self.num_classes)

        outputs['logits'] = x
        tf.logging.info('last conv shape %s', x.get_shape())

        with tf.variable_scope('pred'):
            y_prob = tf.nn.softmax(x)
            outputs['y_prob'] = y_prob
            y_ = tf.argmax(x, axis=-1)
            outputs['y_'] = y_

        return outputs