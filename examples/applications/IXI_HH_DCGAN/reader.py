from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import SimpleITK as sitk
import tensorflow as tf
import os
import numpy as np

from dltk.io.augmentation import extract_random_example_array
from dltk.io.preprocessing import normalise_one_one
import scipy


def read_fn(file_references, mode, params=None):
    """A custom python read function for interfacing with nii image files.

    Args:
        file_references (list): A list of lists containing file references,
            such as [['id_0', 'image_filename_0', target_value_0], ...,
             ['id_N', 'image_filename_N', target_value_N]].
        mode (str): One of the tf.estimator.ModeKeys strings: TRAIN, EVAL or
            PREDICT.
        params (dict, optional): A dictionary to parametrise read_fn ouputs
            (e.g. reader_params = {'n_examples': 10, 'example_size':
            [64, 64, 64], 'extract_examples': True}, etc.).

    Yields:
        dict: A dictionary of reader outputs for dltk.io.abstract_reader.
    """

    for f in file_references:
        subject_id = f[0]

        data_path = '../../../data/IXI_HH/1mm'

        # Read the image nii with sitk
        t1_fn = os.path.join(data_path, '{}/T1_1mm.nii.gz'.format(subject_id))
        t1 = sitk.GetArrayFromImage(sitk.ReadImage(str(t1_fn)))

        # Normalise volume images
        t1 = t1[..., np.newaxis]

        # restrict to slices around center slice
        t1 = t1[len(t1) // 2 - 5:len(t1) // 2 + 5]

        t1 = normalise_one_one(t1)

        images = t1

        noise = np.random.normal(size=(1, 1, 1, 100))

        if mode == tf.estimator.ModeKeys.PREDICT:
            yield {'labels': images, 'features': {'noise': noise}}

        # Check if the reader is supposed to return training examples or full
        # images
        if params['extract_examples']:
            images = extract_random_example_array(
                image_list=images,
                example_size=params['example_size'],
                n_examples=params['n_examples'])

            for e in range(params['n_examples']):
                zoomed = scipy.ndimage.zoom(
                    images[e], (1, 64. / 224., 64. / 224., 1)).astype(
                    np.float32)
                yield {'labels': zoomed,
                       'features': {'noise': noise}}
        else:
            yield {'labels': images, 'features': {'noise': noise}}

    return
