# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from builtins import input

import argparse
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

from dltk.core.metrics import *
from dltk.core.losses import *
from dltk.models.gan.dcgan import dcgan_discriminator_3D, dcgan_generator_3D
from dltk.io.abstract_reader import Reader

from reader import read_fn

BATCH_SIZE = 8
MAX_STEPS = 35000


def train(args):

    np.random.seed(42)
    tf.set_random_seed(42)

    print('Setting up...')

    # Parse csv files for file names
    all_filenames = pd.read_csv(args.data_csv, dtype=object, keep_default_na=False, na_values=[]).as_matrix()
    
    train_filenames = all_filenames[:100]
    val_filenames = all_filenames[100:]
    
    # Set up a data reader to handle the file i/o. 
    reader_params = {'n_examples': 10, 'example_size': [4, 224, 224], 'extract_examples': True}
    reader_example_shapes = {'labels': [4, 64, 64, 1],
                             'features': {'noise': [1, 1, 1, 100]}}
    reader = Reader(read_fn, {'labels': tf.float32, 'features': {'noise': tf.float32}})

    # Get input functions and queue initialisation hooks for training and validation data
    train_input_fn, train_qinit_hook = reader.get_inputs(train_filenames, tf.estimator.ModeKeys.TRAIN,
                                                         example_shapes=reader_example_shapes,
                                                         batch_size=BATCH_SIZE, params=reader_params)

    tfgan = tf.contrib.gan

    # See TFGAN's `train.py` for a description of the generator and
    # discriminator API.
    def generator_fn(generator_inputs):
        gen = dcgan_generator_3D(generator_inputs['noise'], 1, num_convolutions=2, filters=(256, 128, 64, 32, 16),
                                 strides=((4, 4, 4), (1, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2)),
                                 mode=tf.estimator.ModeKeys.TRAIN)
        gen = gen['gen']
        gen = tf.nn.sigmoid(gen)
        tf.summary.image('pred', gen[:, 0])
        return gen

    def discriminator_fn(data, conditioning):
        tf.summary.image('data', data[:, 0])
        disc = dcgan_discriminator_3D(data, filters=(32, 64, 128, 256),
                                           strides=((1, 2, 2), (2, 2, 2), (2, 2, 2), (1, 2, 2)),
                                           mode=tf.estimator.ModeKeys.TRAIN)
        return disc['logits']
    
    # Hooks for training and validation summaries
    step_cnt_hook = tf.train.StepCounterHook(output_dir=args.save_path)

    # Create GAN estimator.
    gan_estimator = tfgan.estimator.GANEstimator(
        args.save_path,
        generator_fn=generator_fn,
        discriminator_fn=discriminator_fn,
        generator_loss_fn=tfgan.losses.least_squares_generator_loss,
        discriminator_loss_fn=tfgan.losses.least_squares_discriminator_loss,
        generator_optimizer=tf.train.AdamOptimizer(0.0005, 0.5, epsilon=1e-5),
        discriminator_optimizer=tf.train.AdamOptimizer(0.0005, 0.5, epsilon=1e-5))
    
    print('Starting training...')
    try:
        gan_estimator.train(train_input_fn, hooks=[train_qinit_hook, step_cnt_hook], steps=MAX_STEPS)

    except KeyboardInterrupt:
        pass
    print('Stopping now.')
    export_dir = gan_estimator.export_savedmodel(
        export_dir_base=args.save_path,
        serving_input_receiver_fn=reader.serving_input_receiver_fn(reader_example_shapes))
    print('Model saved to {}.'.format(export_dir))
        
if __name__ == '__main__':

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Example: IXI HH LSGAN training script')
    parser.add_argument('--run_validation', default=True)
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--cuda_devices', '-c', default='0')
    
    parser.add_argument('--save_path', '-p', default='/tmp/IXI_dcgan/')
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
    
    # Create model save path
    os.system("rm -rf %s" % args.save_path)
    os.system("mkdir -p %s" % args.save_path)

    # Call training
    train(args)