from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
import numpy as np


def sparse_balanced_crossentropy(logits, labels):
    """
    Calculates a class frequency balanced crossentropy loss from sparse labels.

    Args:
        logits (tf.Tensor): logits prediction for which to calculate
            crossentropy error
        labels (tf.Tensor): sparse labels used for crossentropy error
            calculation

    Returns:
        tf.Tensor: Tensor scalar representing the mean loss
    """

    epsilon = tf.constant(np.finfo(np.float32).tiny)

    num_classes = tf.cast(tf.shape(logits)[-1], tf.int32)

    probs = tf.nn.softmax(logits)
    probs += tf.cast(tf.less(probs, epsilon), tf.float32) * epsilon
    log = -1. * tf.log(probs)

    onehot_labels = tf.one_hot(labels, num_classes)

    class_frequencies = tf.stop_gradient(tf.bincount(
        labels, minlength=num_classes, dtype=tf.float32))

    weights = (1. / (class_frequencies + tf.constant(1e-8)))
    weights *= (tf.cast(tf.reduce_prod(tf.shape(labels)), tf.float32) / tf.cast(num_classes, tf.float32))

    new_shape = (([1, ] * len(labels.get_shape().as_list())) + [logits.get_shape().as_list()[-1]])

    weights = tf.reshape(weights, new_shape)

    loss = tf.reduce_mean(tf.reduce_sum(onehot_labels * log * weights, axis=-1))

    return loss


def dice_loss(logits,
              labels,
              num_classes,
              smooth=1e-5,
              include_background=True,
              only_present=False):
    """Calculates a smooth Dice coefficient loss from sparse labels.

    Args:
        logits (tf.Tensor): logits prediction for which to calculate
            crossentropy error
        labels (tf.Tensor): sparse labels used for crossentropy error
            calculation
        num_classes (int): number of class labels to evaluate on
        smooth (float): smoothing coefficient for the loss computation
        include_background (bool): flag to include a loss on the background
            label or not
        only_present (bool): flag to include only labels present in the
            inputs or not

    Returns:
        tf.Tensor: Tensor scalar representing the loss
    """

    # Get a softmax probability of the logits predictions and a one hot
    # encoding of the labels tensor
    probs = tf.nn.softmax(logits)
    onehot_labels = tf.one_hot(
        indices=labels,
        depth=num_classes,
        dtype=tf.float32,
        name='onehot_labels')

    # Compute the Dice similarity coefficient
    label_sum = tf.reduce_sum(onehot_labels, axis=[1, 2, 3], name='label_sum')
    pred_sum = tf.reduce_sum(probs, axis=[1, 2, 3], name='pred_sum')
    intersection = tf.reduce_sum(onehot_labels * probs, axis=[1, 2, 3],
                                 name='intersection')

    per_sample_per_class_dice = (2. * intersection + smooth)
    per_sample_per_class_dice /= (label_sum + pred_sum + smooth)

    # Include or exclude the background label for the computation
    if include_background:
        flat_per_sample_per_class_dice = tf.reshape(
            per_sample_per_class_dice, (-1, ))
        flat_label = tf.reshape(label_sum, (-1, ))
    else:
        flat_per_sample_per_class_dice = tf.reshape(
            per_sample_per_class_dice[:, 1:], (-1, ))
        flat_label = tf.reshape(label_sum[:, 1:], (-1, ))

    # Include or exclude non-present labels for the computation
    if only_present:
        masked_dice = tf.boolean_mask(flat_per_sample_per_class_dice,
                                      tf.logical_not(tf.equal(flat_label, 0)))
    else:
        masked_dice = tf.boolean_mask(
            flat_per_sample_per_class_dice,
            tf.logical_not(tf.is_nan(flat_per_sample_per_class_dice)))

    dice = tf.reduce_mean(masked_dice)
    loss = 1. - dice

    return loss
