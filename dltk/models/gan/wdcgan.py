from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf

from dltk.core.modules import *
from dltk.models.gan.dcgan import DCGAN


class WDCGAN(DCGAN):
    """Deep convolution generative network trained with a wasserstein loss
    based on https://github.com/igul222/improved_wgan_training

    """
    def __init__(self, discriminator_filters=(64, 128, 256, 512), generator_filters=(512, 256, 128, 64, 1),
                 discriminator_strides=((1, 1, 1), (2, 2, 2), (1, 1, 1), (2, 2, 2)),
                 generator_strides=((7, 7, 7), (2, 2, 2), (2, 2, 2)), relu_leakiness=0.01,
                 generator_activation=tf.identity, clip_val=0.01, improved=True, name='wdcgan'):
        """Wasserstein Deep Convolutional Generative Adversarial Network

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
        clip_val: float
            If improved, factor for the gradient penalty. If not improved, value to clip the weights by
        improved : bool
            flag to toggle improved wasserstein GAN
        name : string
            name of the network used for scoping
        """
        self.batch_size = tf.placeholder(tf.int32, shape=[])
        self.improved = improved
        self.clip_val = clip_val

        super(WDCGAN, self).__init__(discriminator_filters, generator_filters, discriminator_strides,
                                     generator_strides, relu_leakiness, generator_activation, name)

    def _build(self, noise, samples, is_training=True):
        """Constructs a Wasserstein GAN

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
                - `disc_gen_logits` - mean of logit values of discriminator on generated samples
                - `disc_sample_logits` - mean of logit values of discriminator on real samples
                - `disc_sample` - discriminator output dictionary for real sample
                - `d_loss` - discriminator loss
                - `g_loss` - generator loss
                - `clip_ops` - None if improved wgan, else operation to run to clip critic/discriminator

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

        out['disc_gen_logits'] = tf.reduce_mean(out['disc_gen']['logits'])
        out['disc_sample_logits'] = tf.reduce_mean(out['disc_sample']['logits'])

        out['d_loss'] = out['disc_gen_logits'] - out['disc_sample_logits']
        out['g_loss'] = - out['disc_gen_logits']

        if self.improved:
            with tf.variable_scope('gradient_penalty'):
                alpha = tf.random_uniform(
                    shape=[self.batch_size,1,1,1],
                    minval=0.,
                    maxval=1.
                )
                differences = out['gen']['gen'] - samples
                interpolates = samples + alpha * differences
                gradients = tf.gradients(self.disc(interpolates)['logits'], [interpolates])[0]
                slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
                gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
                # Over loading clip_val
                out['d_loss'] += self.clip_val * gradient_penalty
                out['clip_ops'] = None
        else:
            with tf.variable_scope('clipping'):
                clip_ops = []
                for var in self.disc.get_variables():
                    clip_ops.append(
                        tf.assign(
                            var,
                            tf.clip_by_value(var, -self.clip_val, self.clip_val)
                        )
                    )
                out['clip_ops'] = tf.group(*clip_ops)


        return out