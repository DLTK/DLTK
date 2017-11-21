# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import argparse
import os
import pandas as pd
import tensorflow as tf
import numpy as np

from dltk.networks.gan.dcgan import dcgan_discriminator_3d, dcgan_generator_3d
from dltk.io.abstract_reader import Reader

from reader import read_fn

BATCH_SIZE = 8
MAX_STEPS = 35000
SAVE_SUMMARY_STEPS = 100


def train(args):
    np.random.seed(42)
    tf.set_random_seed(42)

    print('Setting up...')

    # Parse csv files for file names
    all_filenames = pd.read_csv(
        args.data_csv,
        dtype=object,
        keep_default_na=False,
        na_values=[]).as_matrix()

    train_filenames = all_filenames

    # Set up a data reader to handle the file i/o.
    reader_params = {'n_examples': 10,
                     'example_size': [4, 224, 224],
                     'extract_examples': True}

    reader_example_shapes = {'labels': [4, 64, 64, 1],
                             'features': {'noise': [1, 1, 1, 100]}}

    reader = Reader(read_fn, {'features': {'noise': tf.float32},
                              'labels': tf.float32})

    # Get input functions and queue initialisation hooks for data
    train_input_fn, train_qinit_hook = reader.get_inputs(
        file_references=train_filenames,
        mode=tf.estimator.ModeKeys.TRAIN,
        example_shapes=reader_example_shapes,
        batch_size=BATCH_SIZE,
        params=reader_params)

    # See TFGAN's `train.py` for a description of the generator and
    # discriminator API.
    def generator_fn(generator_inputs):
        """Generator function to build fake data samples. It creates a network
        given input features (e.g. from a dltk.io.abstract_reader). Further,
        custom Tensorboard summary ops can be added. For additional
        information, please refer to https://www.tensorflow.org/versions/master/api_docs/python/tf/contrib/gan/estimator/GANEstimator.

        Args:
            generator_inputs (tf.Tensor): Noise input to generate samples from.

        Returns:
            tf.Tensor: Generated data samples
        """
        gen = dcgan_generator_3d(
            inputs=generator_inputs['noise'],
            mode=tf.estimator.ModeKeys.TRAIN)
        gen = gen['gen']
        gen = tf.nn.tanh(gen)
        return gen

    def discriminator_fn(data, conditioning):
        """Discriminator function to discriminate real and fake data. It creates
        a network given input features (e.g. from a dltk.io.abstract_reader).
        Further, custom Tensorboard summary ops can be added. For additional
        information, please refer to https://www.tensorflow.org/versions/master/api_docs/python/tf/contrib/gan/estimator/GANEstimator.

        Args:
            generator_inputs (tf.Tensor): Noise input to generate samples from.

        Returns:
            tf.Tensor: Generated data samples
        """
        tf.summary.image('data', data[:, 0])

        disc = dcgan_discriminator_3d(
            inputs=data,
            mode=tf.estimator.ModeKeys.TRAIN)

        return disc['logits']

    # get input tensors from queue
    features, labels = train_input_fn()

    # build generator
    with tf.variable_scope('generator'):
        gen = generator_fn(features)

    # build discriminator on fake data
    with tf.variable_scope('discriminator'):
        disc_fake = discriminator_fn(gen, None)

    # build discriminator on real data, reusing the previously created variables
    with tf.variable_scope('discriminator', reuse=True):
        disc_real = discriminator_fn(labels, None)

    # building an LSGAN loss for the real examples
    d_loss_real = tf.losses.mean_squared_error(
        disc_real, tf.ones_like(disc_real))

    # calculating a pseudo accuracy for the discriminator detecting a real
    # sample and logging that
    d_pred_real = tf.cast(tf.greater(disc_real, 0.5), tf.float32)
    _, d_acc_real = tf.metrics.accuracy(tf.ones_like(disc_real), d_pred_real)
    tf.summary.scalar('disc/real_acc', d_acc_real)

    # building an LSGAN loss for the fake examples
    d_loss_fake = tf.losses.mean_squared_error(
        disc_fake, tf.zeros_like(disc_fake))

    # calculating a pseudo accuracy for the discriminator detecting a fake
    # sample and logging that
    d_pred_fake = tf.cast(tf.greater(disc_fake, 0.5), tf.float32)
    _, d_acc_fake = tf.metrics.accuracy(tf.zeros_like(disc_fake), d_pred_fake)
    tf.summary.scalar('disc/fake_acc', d_acc_fake)

    # building an LSGAN loss for the generator
    g_loss = tf.losses.mean_squared_error(
        disc_fake, tf.ones_like(disc_fake))
    tf.summary.scalar('loss/gen', g_loss)

    # combining the discriminator losses
    d_loss = d_loss_fake + d_loss_real
    tf.summary.scalar('loss/disc', d_loss)

    # getting the list of discriminator variables
    d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                               'discriminator')

    # building the discriminator optimizer
    d_opt = tf.train.AdamOptimizer(
        0.001, 0.5, epsilon=1e-5).minimize(d_loss, var_list=d_vars)

    # getting the list of generator variables
    g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                               'generator')

    # building the generator optimizer
    g_opt = tf.train.AdamOptimizer(
        0.001, 0.5, epsilon=1e-5).minimize(g_loss, var_list=g_vars)

    # getting a variable to hold the global step
    global_step = tf.train.get_or_create_global_step()
    # build op to increment the global step - important for TensorBoard logging
    inc_step = global_step.assign_add(1)

    # build the training session.
    # NOTE: we are not using a tf.estimator here, because they prevent some
    # flexibility in the training procedure
    s = tf.train.MonitoredTrainingSession(checkpoint_dir=args.model_path,
                                          save_summaries_steps=100,
                                          save_summaries_secs=None,
                                          hooks=[train_qinit_hook])

    # build dummy logging string
    log = 'Step {} with Loss D: {}, Loss G: {}, Acc Real: {} Acc Fake: {}'

    # start training
    print('Starting training...')
    loss_d = 0
    loss_g = 0
    try:
        for step in range(MAX_STEPS):
            # if discriminator is too good, only train generator
            if not loss_g > 3 * loss_d:
                s.run(d_opt)

            # if generator is too good, only train discriminator
            if not loss_d > 3 * loss_g:
                s.run(g_opt)

            # increment global step for logging hooks
            s.run(inc_step)

            # get statistics for training scheduling
            loss_d, loss_g, acc_d, acc_g = s.run(
                [d_loss, g_loss, d_acc_real, d_acc_fake])

            # print stats for information
            if step % SAVE_SUMMARY_STEPS == 0:
                print(log.format(step, loss_d, loss_g, acc_d, acc_g))
    except KeyboardInterrupt:
        pass
    print('Stopping now.')


if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Example: IXI HH LSGAN training script')
    parser.add_argument('--run_validation', default=True)
    parser.add_argument('--restart', default=False, action='store_true')
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--cuda_devices', '-c', default='0')

    parser.add_argument('--model_path', '-p', default='/tmp/IXI_dcgan/')
    parser.add_argument('--data_csv', default='../../../data/IXI_HH/demographic_HH.csv')

    args = parser.parse_args()

    # Set verbosity
    if args.verbose:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        tf.logging.set_verbosity(tf.logging.INFO)
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf.logging.set_verbosity(tf.logging.ERROR)

    # GPU allocation options
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    # Handle restarting and resuming training
    if args.restart:
        print('Restarting training from scratch.')
        os.system('rm -rf {}'.format(args.model_path))

    if not os.path.isdir(args.model_path):
        os.system('mkdir -p {}'.format(args.model_path))
    else:
        print('Resuming training on model_path {}'.format(args.model_path))

    # Call training
    train(args)
