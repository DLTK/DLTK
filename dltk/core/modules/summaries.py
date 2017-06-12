from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO


def image_summary(img, summary_name, collections=None):
    """Builds an image summary from a tf.Tensor or np.ndarray

    If the image is a tf.Tensor 4D and 5D tensors of form (batch, x, y, channels) and (batch, x, y, z, channels) are
    supported. For 5D tensors each middle slice is plotted if the size of the tensor is known. Otherwise the first
    slice is taken.

    If the image is a np.ndarray 3D and 4D arrays of form (x, y, channels) and (x, y, z, channels) are supported. For
    4D tensors each middle slice is plotted if the size of the tensor is known. Otherwise the first slice is taken.

    Parameters
    ----------
    img : tf.Tensor or np.ndarray
        image to be plotted
    summary_name : string
        name of the summary to be produced
    collections : list or tuple, optional
        list of collections this summary should be added to additionally to `tf.GraphKeys.SUMMARIES` and
        `image_summaries`

    Returns
    -------
    tf.Tensor or tf.Summary
        Tensor produced from tf.summary or Summary object with the plotted image(s)

    """
    summaries = []
    if isinstance(img, tf.Tensor):
        collections = [tf.GraphKeys.SUMMARIES, 'image_summaries'] + collections if collections is not None else []
        if len(img.get_shape().as_list()) == 5:
            for dim in range(3):
                slicer = [slice(None)] * 4
                pos = 0
                if img.get_shape().as_list()[dim + 1]:
                    pos = img.get_shape().as_list()[dim + 1] // 2
                slicer[dim + 1] = pos

                summaries.append(tf.summary.image('{}_dim{}'.format(summary_name, dim), img[slicer],
                                                  collections=collections))
        else:
            summaries.append(tf.summary.image(summary_name, img, collections=collections))
        return tf.summary.merge(summaries)
    elif isinstance(img, np.ndarray):
        # only works on 3D and 4D arrays -> batch isn't used
        # see https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514#file-tensorboard_logging-py-L41
        if np.min(img) < 0.:
            img -= np.min(img)
            img /= np.max(img)

        if img.ndim == 4:
            for dim in range(3):
                slicer = [slice(None)] * 3
                slicer[dim] = img.shape[dim] // 2

                tmp_img = (img[slicer] + img[slicer].min())
                tmp_img /= tmp_img.max()

                s = StringIO()
                if img.shape[-1] == 1:
                    plt.imsave(s, img[slicer][:, :, 0], format='png', cmap='gray')
                else:
                    plt.imsave(s, img[slicer], format='png')

                # Create an Image object
                img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                           height=img[slicer].shape[0],
                                           width=img[slicer].shape[1])
                # Create a Summary value
                summaries.append(tf.Summary.Value(tag='{}_dim{}'.format(summary_name, dim),
                                                  image=img_sum))
        else:
            s = StringIO()
            if img.shape[-1] == 1:
                plt.imsave(s, img[:, :, 0], format='png', cmap='gray')
            else:
                plt.imsave(s, img, format='png')

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            summaries.append(tf.Summary.Value(tag=summary_name,
                                              image=img_sum))
        return tf.Summary(value=summaries)
    else:
        raise Exception('Only tf.Tensors and np.ndarrays are supported.')


def scalar_summary(x, summary_name, collections=None):
    """Builds a scalar summary

    If x is a tf.Tensor it creates the summary operation to track x

    If x is a scalar it creates the tf.Summary object to be written be a summary writer

    If x is a list, tuple or dict a tf.Summary object is created for each element. The key or index is used for naming

    Parameters
    ----------
    x : tf.Tensor or scalar or list or dict
        scalar data to be plotted
    summary_name : string
        name of the summary to be produced
    collections : list or tuple, optional
        list of collections this summary should be added to additionally to `tf.GraphKeys.SUMMARIES` and
        `image_summaries`

    Returns
    -------
    tf.Tensor or tf.Summary
        Tensor produced from tf.summary or Summary object with the summarised data

    """
    if isinstance(x, tf.Tensor):
        collections = [tf.GraphKeys.SUMMARIES, 'scalar_summaries'] + collections if collections is not None else []
        return tf.summary.scalar(summary_name, x, collections)
    elif np.isscalar(x):
        return tf.Summary(value=[tf.Summary.Value(tag=summary_name, simple_value=x)])
    elif isinstance(x, (list, tuple)):
        return tf.Summary(value=[tf.Summary.Value(tag='{}_{}'.format(summary_name, i), simple_value=xi)
                                 for i, xi in enumerate(x)])
    elif isinstance(x, dict):
        return tf.Summary(value=[tf.Summary.Value(tag='{}_{}'.format(summary_name, i), simple_value=xi)
                                 for i, xi in x.items()])
    else:
        raise Exception('Only tf.Tensors and np.ndarrays are supported.')
