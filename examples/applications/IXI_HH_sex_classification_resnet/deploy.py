# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import argparse
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.contrib import predictor

from reader import read_fn

READER_PARAMS = {'extract_examples': False}
N_VALIDATION_SUBJECTS = 28


def predict(args):
    # Read in the csv with the file names you would want to predict on
    file_names = pd.read_csv(
        args.csv,
        dtype=object,
        keep_default_na=False,
        na_values=[]).as_matrix()

    # We trained on the first 4 subjects, so we predict on the rest
    file_names = file_names[-N_VALIDATION_SUBJECTS:]

    # From the model save_path, parse the latest saved model and restore a
    # predictor from it
    export_dir = \
        [os.path.join(args.model_path, o) for o in os.listdir(args.model_path)
         if os.path.isdir(os.path.join(args.model_path, o))
         and o.isdigit()][-1]
    print('Loading from {}'.format(export_dir))
    my_predictor = predictor.from_saved_model(export_dir)

    # Iterate through the files, predict on the full volumes and compute a Dice
    # coefficient
    accuracy = []
    for output in read_fn(file_references=file_names,
                          mode=tf.estimator.ModeKeys.EVAL,
                          params=READER_PARAMS):
        t0 = time.time()

        # Parse the read function output and add a dummy batch dimension as
        # required
        img = np.expand_dims(output['features']['x'], axis=0)
        lbl = np.expand_dims(output['labels']['y'], axis=0)

        print('Running image with shape {}. '.format(img.shape))

        # Do a sliding window inference with our DLTK wrapper
        y_ = my_predictor.session.run(
            fetches=my_predictor._fetch_tensors['y_'],
            feed_dict={my_predictor._feed_tensors['x']: img})

        # Calculate the accuracy for this subject
        accuracy.append(y_ == lbl)

        # Print outputs
        print('pred={}; true={}; time={}; '.format(
            y_, lbl, time.time() - t0))
    print('accuracy={}'.format(np.mean(accuracy)))


if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='IXI HH example sex classification deploy script')
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--cuda_devices', '-c', default='0')

    parser.add_argument('--model_path', '-p',
                        default='/tmp/IXI_sex_classification/')
    parser.add_argument('--csv', default='../../../data/IXI_HH/demographic_HH.csv')

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

    # Call training
    predict(args)
