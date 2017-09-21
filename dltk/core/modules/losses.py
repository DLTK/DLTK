from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
from dltk.core.modules.summaries import *
import numpy as np


def sparse_crossentropy(logits, labels, name='crossentropy', collections=['losses']):
    """ Crossentropy loss

    Calculates the crossentropy loss and builds a scalar summary.

    Parameters
    ----------
    logits : tf.Tensor
        logit prediction for which to calculate crossentropy error
    labels : tf.Tensor
        labels used for crossentropy error calculation
    name : string
        name of this operation and summary
    collections : list or tuple
        list of collections to add the summaries to

    Returns
    -------
    tf.Tensor
        Tensor representing the loss

    """
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    loss = tf.reduce_mean(ce, name=name)
    scalar_summary(loss, name, collections)
    return loss

def sparse_balanced_crossentropy(logits, labels, name='crossentropy', collections=['losses']):
    """ Crossentropy loss

    Calculates the crossentropy loss and builds a scalar summary.

    Parameters
    ----------
    logits : tf.Tensor
        logit prediction for which to calculate crossentropy error
    labels : tf.Tensor
        labels used for crossentropy error calculation
    name : string
        name of this operation and summary
    collections : list or tuple
        list of collections to add the summaries to

    Returns
    -------
    tf.Tensor
        Tensor representing the loss

    """
    eps = tf.constant(np.finfo(np.float32).tiny)

    num_classes = tf.cast(tf.shape(logits)[-1], tf.int32)

    probs = tf.nn.softmax(logits)
    probs += tf.cast(tf.less(probs, eps), tf.float32) * eps
    log = -1. * tf.log(probs)

    oh_labels = tf.one_hot(labels, num_classes)

    class_occurances = tf.stop_gradient(tf.bincount(labels, minlength=num_classes, dtype=tf.float32))
    weights = (1. / (class_occurances + tf.constant(1e-8))) * (tf.cast(tf.reduce_prod(tf.shape(labels)), tf.float32)
                                                  / tf.cast(num_classes, tf.float32))
    weights = tf.reshape(weights, ([1,] * len(labels.get_shape().as_list())) + [logits.get_shape().as_list()[-1]])



    loss = tf.reduce_mean(tf.reduce_sum(oh_labels * log * weights, axis=-1))

    scalar_summary(loss, name, collections)
    return loss


def mse(x, y, name='mse', collections=['losses']):
    """ Mean squared error

        Calculates the crossentropy loss and builds a scalar summary.

        Parameters
        ----------
        x : tf.Tensor
            prediction for which to calculate the error
        y : tf.Tensor
            targets with which to calculate the error
        name : string
            name of this operation and summary
        collections : list or tuple
            list of collections to add the summaries to

        Returns
        -------
        tf.Tensor
            Tensor representing the loss

        """
    loss = tf.reduce_mean(tf.square(x - y), name=name)
    scalar_summary(loss, name, collections)
    return loss


def dice_loss(logits, labels, num_classes, smooth=1e-5, include_background=True, only_present=False,
              name='dice_loss', collections=['losses']):
    """ Smooth dice loss

        Calculates the smooth dice loss and builds a scalar summary.

        Parameters
        ----------
        logits : tf.Tensor
            prediction for which to calculate the error
        labels : tf.Tensor
            sparse targets with which to calculate the error
        num_classes : int 
            number of class labels to evaluate on
        include_background : bool
            flag to include a loss on the background label or not
        name : string
            name of this operation and summary
        collections : list or tuple
            list of collections to add the summaries to

        Returns
        -------
        tf.Tensor
            Tensor representing the loss

    """
    
    probs = tf.nn.softmax(logits)
    onehot_labels = tf.one_hot(labels, num_classes, dtype=tf.float32, name='onehot_labels')
    
    label_sum = tf.reduce_sum(onehot_labels, axis=[1, 2, 3], name='label_sum')

    pred_sum = tf.reduce_sum(probs, axis=[1, 2, 3], name='pred_sum')
    
    intersection = tf.reduce_sum(onehot_labels * probs, axis=[1, 2, 3], name='intersection')

    per_sample_per_class_dice = (2. * intersection + smooth) / (label_sum + pred_sum + smooth)

    flat_per_sample_per_class_dice = tf.reshape(per_sample_per_class_dice if include_background
                                                else per_sample_per_class_dice[:, 1:] , (-1, ))

    if only_present:
        flat_label = tf.reshape(label_sum if include_background
                                else label_sum[:, 1:] , (-1, ))
        masked_dice = tf.boolean_mask(flat_per_sample_per_class_dice,
                                      tf.logical_not(tf.equal(flat_label, 0)))
    else:
        masked_dice = tf.boolean_mask(flat_per_sample_per_class_dice,
                                      tf.logical_not(tf.is_nan(flat_per_sample_per_class_dice)))

    dice = tf.reduce_mean(masked_dice)

    loss = 1. - dice
    
    scalar_summary(loss, name, collections)

    return loss