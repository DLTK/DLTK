from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages

from dltk.core.modules import *

class DCGAN(AbstractModule):
    """Convolutional Autoencoder

    This module builds a convolutional autoencoder with varying number of layers and hidden units.
    """
    def __init__(self, discriminator_filters=(64, 128, 256, 512), generator_filters=(512, 256, 128, 64, 1),
                 discriminator_strides=((1, 1, 1), (2, 2, 2), (1, 1, 1), (2, 2, 2)),
                 generator_strides=((7, 7, 7), (2, 2, 2), (2, 2, 2)), relu_leakiness=0.01,
                 generator_activation=tf.identity, name='dcgan'):
        """Deep Convolutional Generative Adversarial Network

        Parameters
        ----------
        discriminator_filters : list or tuple
            list of filters used for the discriminator
        generator_filters : list or tuple
            list of filters used for the generator
        discriminator_strides : list or tuple
            list of strides used for the discriminator
        generator_strides : list or tuple
            list of strides used for the generator
        relu_leakiness : float
            leakiness of the relus used in the discriminator
        generator_activation : function
            function to be used as activation for the generator
        name : string
            name of the network used for scoping
        """
        self.discriminator_filters = discriminator_filters
        self.discriminator_strides = discriminator_strides
        self.generator_filters = generator_filters
        self.generator_strides = generator_strides
        self.discriminator_strides = discriminator_strides
        self.relu_leakiness = relu_leakiness
        self.generator_activation = generator_activation
        self.in_filter = None

        assert (len(discriminator_filters) == len(discriminator_strides))

        super(DCGAN, self).__init__(name)

    class Discriminator(AbstractModule):
        def __init__(self, filters, strides, relu_leakiness, name):
            """Constructs the discriminator of a DCGAN

            Parameters
            ----------
            filters : list or tuple
                filters for convolutional layers
            strides : list or tuple
                strides to be used for convolutions
            relu_leakiness : float
                leakines of relu nonlinearity
            name : string
                name of the network
            """
            self.filters = filters
            self.strides = strides
            self.relu_leakiness = relu_leakiness
            self.in_filter = None

            assert (len(strides) == len(filters))

            super(DCGAN.Discriminator, self).__init__(name)

        def _build(self, x, is_training=True):
            if self.in_filter is None:
                self.in_filter = x.get_shape().as_list()[-1]
            assert (self.in_filter == x.get_shape().as_list()[-1], 'Network was built for a different input shape')

            out = {}
            for i in range(len(self.filters) -1):
                with tf.variable_scope('l{}'.format(i)):
                    x = Convolution(self.filters[i], 4, self.strides[i])(x)
                    x = BatchNorm()(x)
                    x = leaky_relu(x, self.relu_leakiness)

            with tf.variable_scope('final'):
                x = tf.reshape(x, (tf.shape(x)[0], np.prod(x.get_shape().as_list()[1:])))
                x = Linear(1)(x)
                out['logits'] = x
                x = tf.nn.sigmoid(x)
                out['probs'] = x
                out['pred'] = tf.greater(x, 0.5)

            return out

    class Generator(AbstractModule):
        def __init__(self, filters, strides, output_activation, name):
            """Constructs the discriminator of a DCGAN

            Parameters
            ----------
            filters : list or tuple
                filters for convolutional layers
            strides : list or tuple
                strides to be used for convolutions
            name : string
                name of the network
            """
            self.filters = filters
            self.strides = strides
            self.in_filter = None
            self.output_activation = output_activation
            assert (len(strides) == len(filters))

            super(DCGAN.Generator, self).__init__(name)

        def _build(self, x, is_training=True):
            if self.in_filter is None:
                self.in_filter = x.get_shape().as_list()[-1]
            assert (self.in_filter == x.get_shape().as_list()[-1], 'Network was built for a different input shape')

            x = tf.reshape(x, [tf.shape(x)[0]] + [1,] * len(self.strides[0]) + [self.in_filter])

            out = {}
            for i in range(len(self.filters) - 1):
                with tf.variable_scope('l{}'.format(i)):
                    x = TransposedConvolution(self.filters[i], strides=self.strides[i])(x)
                    x = BatchNorm()(x)
                    x = tf.nn.relu(x)

            with tf.variable_scope('final'):
                x = TransposedConvolution(self.filters[-1], strides=self.strides[-1])(x)
                x = self.output_activation(x)
                out['gen'] = x

            return out

    def _build(self, noise, samples, is_training=True):
        """Constructs a DCGAN

        Parameters
        ----------
        noise : tf.Tensor
            noise tensor for the generator to generate fake samples
        samples : tf.Tensor
            real samples used by the discriminator
        is_training : bool
            flag to specify whether this is training - passed to batch normalization

        Returns
        -------
        dict
            output dictionary containing:
                - `gen` - generator output dictionary
                    - `gen` - generated sample
                - `disc_gen` - discriminator output dictionary for generated sample
                - `disc_sample` - discriminator output dictionary for real sample
                - `d_loss` - discriminator loss
                - `g_loss` - generator loss

        """

        if self.in_filter is None:
            self.in_filter = samples.get_shape().as_list()[-1]
        assert(self.in_filter == samples.get_shape().as_list()[-1], 'Network was built for a different input shape')
        assert(self.in_filter == self.generator_filters[-1], 'Generator was built for a different sample shape')

        out = {}

        self.disc = self.Discriminator(self.discriminator_filters, self.discriminator_strides, self.relu_leakiness,
                                       'disc')
        self.gen = self.Generator(self.generator_filters, self.generator_strides, self.generator_activation, 'gen')

        out['gen'] = self.gen(noise)
        out['disc_gen'] = self.disc(out['gen']['gen'])
        out['disc_sample'] = self.disc(samples)

        out['d_loss'] = -(tf.reduce_mean(tf.log(out['disc_sample']['probs']))
                         + tf.reduce_mean(tf.log(1. - out['disc_gen']['probs'])))
        out['g_loss'] = -tf.reduce_mean(tf.log(out['disc_gen']['probs']))

        return out