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

from dltk.networks.regression_classification.resnet import resnet_3d
from dltk.io.abstract_reader import Reader

from reader import read_fn


EVAL_EVERY_N_STEPS = 100
EVAL_STEPS = 5

NUM_CLASSES = 2
NUM_CHANNELS = 1

BATCH_SIZE = 8
SHUFFLE_CACHE_SIZE = 32

MAX_STEPS = 50000


def model_fn(features, labels, mode, params):
    """Model function to construct a tf.estimator.EstimatorSpec. It creates a
        network given input features (e.g. from a dltk.io.abstract_reader) and
        training targets (labels). Further, loss, optimiser, evaluation ops and
        custom tensorboard summary ops can be added. For additional information,
         please refer to https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator#model_fn.

    Args:
        features (tf.Tensor): Tensor of input features to train from. Required
            rank and dimensions are determined by the subsequent ops
            (i.e. the network).
        labels (tf.Tensor): Tensor of training targets or labels. Required rank
            and dimensions are determined by the network output.
        mode (str): One of the tf.estimator.ModeKeys: TRAIN, EVAL or PREDICT
        params (dict, optional): A dictionary to parameterise the model_fn
            (e.g. learning_rate)

    Returns:
        tf.estimator.EstimatorSpec: A custom EstimatorSpec for this experiment
    """

    # 1. create a model and its outputs
    net_output_ops = resnet_3d(
        features['x'],
        num_res_units=2,
        num_classes=NUM_CLASSES,
        filters=(16, 32, 64, 128, 256),
        strides=((1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
        mode=mode,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    # 1.1 Generate predictions only (for `ModeKeys.PREDICT`)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=net_output_ops,
            export_outputs={'out': tf.estimator.export.PredictOutput(net_output_ops)})

    # 2. set up a loss function
    one_hot_labels = tf.reshape(tf.one_hot(labels['y'], depth=NUM_CLASSES), [-1, NUM_CLASSES])

    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=one_hot_labels,
        logits=net_output_ops['logits'])

    # 3. define a training op and ops for updating moving averages (i.e. for
    # batch normalisation)
    global_step = tf.train.get_global_step()
    optimiser = tf.train.AdamOptimizer(
        learning_rate=params["learning_rate"],
        epsilon=1e-5)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimiser.minimize(loss, global_step=global_step)

    # 4.1 (optional) create custom image summaries for tensorboard
    my_image_summaries = {}
    my_image_summaries['feat_t1'] = features['x'][0, 32, :, :, 0]

    expected_output_size = [1, 96, 96, 1]  # [B, W, H, C]
    [tf.summary.image(name, tf.reshape(image, expected_output_size))
     for name, image in my_image_summaries.items()]

    # 4.2 (optional) track the rmse (scaled back by 100, see reader.py)
    acc = tf.metrics.accuracy
    prec = tf.metrics.precision
    eval_metric_ops = {"accuracy": acc(labels['y'], net_output_ops['y_']),
                       "precision": prec(labels['y'], net_output_ops['y_'])}

    # 5. Return EstimatorSpec object
    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=net_output_ops,
                                      loss=loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops)


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

    train_filenames = all_filenames[:150]
    val_filenames = all_filenames[150:]

    # Set up a data reader to handle the file i/o.
    reader_params = {'n_examples': 2,
                     'example_size': [64, 96, 96],
                     'extract_examples': True}

    reader_example_shapes = {'features': {'x': reader_params['example_size'] + [NUM_CHANNELS]},
                             'labels': {'y': [1]}}
    reader = Reader(read_fn,
                    {'features': {'x': tf.float32},
                     'labels': {'y': tf.int32}})

    # Get input functions and queue initialisation hooks for training and
    # validation data
    train_input_fn, train_qinit_hook = reader.get_inputs(
        file_references=train_filenames,
        mode=tf.estimator.ModeKeys.TRAIN,
        example_shapes=reader_example_shapes,
        batch_size=BATCH_SIZE,
        shuffle_cache_size=SHUFFLE_CACHE_SIZE,
        params=reader_params)

    val_input_fn, val_qinit_hook = reader.get_inputs(
        file_references=val_filenames,
        mode=tf.estimator.ModeKeys.EVAL,
        example_shapes=reader_example_shapes,
        batch_size=BATCH_SIZE,
        shuffle_cache_size=SHUFFLE_CACHE_SIZE,
        params=reader_params)

    # Instantiate the neural network estimator
    nn = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=args.model_path,
        params={"learning_rate": 0.001},
        config=tf.estimator.RunConfig())

    # Hooks for validation summaries
    val_summary_hook = tf.contrib.training.SummaryAtEndHook(
        os.path.join(args.model_path, 'eval'))
    step_cnt_hook = tf.train.StepCounterHook(every_n_steps=EVAL_EVERY_N_STEPS,
                                             output_dir=args.model_path)

    print('Starting training...')
    try:
        for _ in range(MAX_STEPS // EVAL_EVERY_N_STEPS):
            nn.train(
                input_fn=train_input_fn,
                hooks=[train_qinit_hook, step_cnt_hook],
                steps=EVAL_EVERY_N_STEPS)

            if args.run_validation:
                results_val = nn.evaluate(
                    input_fn=val_input_fn,
                    hooks=[val_qinit_hook, val_summary_hook],
                    steps=EVAL_STEPS)
                print('Step = {}; val loss = {:.5f};'.format(
                    results_val['global_step'],
                    results_val['loss']))

    except KeyboardInterrupt:
        pass

    # When exporting we set the expected input shape to be arbitrary.
    export_dir = nn.export_savedmodel(
        export_dir_base=args.model_path,
        serving_input_receiver_fn=reader.serving_input_receiver_fn(
            {'features': {'x': [None, None, None, NUM_CHANNELS]},
             'labels': {'y': [1]}}))
    print('Model saved to {}.'.format(export_dir))


if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Example: IXI HH resnet sex classification training')
    parser.add_argument('--run_validation', default=True)
    parser.add_argument('--restart', default=False, action='store_true')
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--cuda_devices', '-c', default='0')

    parser.add_argument('--model_path', '-p', default='/tmp/IXI_sex_classification/')
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
