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

from dltk.io.augmentation import extract_random_example_array

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

    # From the model_path, parse the latest saved model and restore a
    # predictor from it
    export_dir = [os.path.join(args.model_path, o) for o in sorted(
        os.listdir(args.model_path)) if os.path.isdir(
        os.path.join(args.model_path, o)) and o.isdigit()][-1]

    print('Loading from {}'.format(export_dir))
    my_predictor = predictor.from_saved_model(export_dir)

    # Iterate through the files, predict on the full volumes and compute a Dice
    # coefficient
    mae = []
    for output in read_fn(file_references=file_names,
                          mode=tf.estimator.ModeKeys.EVAL,
                          params=READER_PARAMS):
        t0 = time.time()

        # Parse the read function output and add a dummy batch dimension as
        # required
        img = output['features']['x']
        lbl = output['labels']['y']
        test_id = output['img_id']

        # We know, that the training input shape of [64, 96, 96] will work with
        # our model strides, so we collect several crops of the test image and
        # average the predictions. Alternatively, we could pad or crop the input
        # to any shape that is compatible with the resolution scales of the
        # model:

        num_crop_predictions = 4
        crop_batch = extract_random_example_array(
            image_list=img,
            example_size=[64, 96, 96],
            n_examples=num_crop_predictions)

        y_ = my_predictor.session.run(
            fetches=my_predictor._fetch_tensors['logits'],
            feed_dict={my_predictor._feed_tensors['x']: crop_batch})

        # Average the predictions on the cropped test inputs:
        y_ = np.mean(y_)

        # Calculate the absolute error for this subject
        mae.append(np.abs(y_ - lbl))

        # Print outputs
        print('id={}; pred={:0.2f} yrs; true={:0.2f} yrs; run time={:0.2f} s; '
              ''.format(test_id, y_, lbl[0], time.time() - t0))
    print('mean absolute err={:0.3f} yrs'.format(np.mean(mae)))


if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='IXI HH example age regression deploy script')
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--cuda_devices', '-c', default='0')

    parser.add_argument('--model_path', '-p',
                        default='/tmp/IXI_age_regression/')
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
