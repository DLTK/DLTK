# -*- coding: utf-8 -*-
#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from builtins import input

import argparse
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
import SimpleITK as sitk

import dltk.core.modules as modules

from dltk.core import metrics as metrics
from dltk.core.io.augmentation import *
from dltk.core.io.preprocessing import *
from dltk.models.segmentation.fcn import ResNetFCN
from tensorflow.contrib import predictor
from dltk.core.utils import sliding_window_segmentation_inference
from dltk.core.metrics import dice

def receiver(filenames):
    for f in filenames[:, 1]:

        img_fn = f

        t1 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(str(img_fn), 'T1.nii')))
        t1_ir = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(str(img_fn), 'T1_IR.nii')))
        t2_fl = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(str(img_fn), 'T2_FLAIR.nii')))
        lbl = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(str(img_fn), 'LabelsForTraining.nii')))

        # normalise volume images
        t1 = whitening(t1)
        t1_ir = whitening(t1_ir)
        t2_fl = whitening(t2_fl)

        # create a 4D multi-sequence image (i.e. [channels,y,x,slice])
        images = np.asarray([t1, t1_ir, t2_fl])

        # transpose to [batch, x,y,z,channel] for learning.
        images = np.transpose(images, (1, 2, 3, 0))


        yield [images, lbl]

    return


def predict(args):
    export_dir = [os.path.join(args.save_path, o) for o in os.listdir(args.save_path)
                  if os.path.isdir(os.path.join(args.save_path, o)) and o.isdigit()][-1]
    print(export_dir)
    pred = predictor.from_saved_model(export_dir)

    print(pred._feed_tensors)
    print(pred._fetch_tensors)
    print(pred.session)

    files = pd.read_csv(args.csv, dtype=object, keep_default_na=False, na_values=[]).as_matrix()
    for image, label in receiver(files):
        image = image[np.newaxis]
        label = label[np.newaxis]

        y_prob = pred._fetch_tensors['y_prob']

        pred = sliding_window_segmentation_inference(pred.session, [y_prob],
                                                     {pred._feed_tensors['x']: image}, batch_size=4)[0]

        pred = np.argmax(pred, -1)

        print('Dice {}'.format(dice(pred, label, y_prob.get_shape().as_list()[-1])[1:].mean()))


    #predictions = predict_fn(
    #    {"x": [[6.4, 3.2, 4.5, 1.5],
    #           [5.8, 3.1, 5.0, 1.7]]})
    #print(predictions['scores'])


if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Example: generic deploy script')
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--cuda_devices', '-c', default='0')

    parser.add_argument('--save_path', '-p', default='/tmp/estimator')
    parser.add_argument('--csv', default='val.csv')

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