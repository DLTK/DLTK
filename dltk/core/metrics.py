from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np


def dice(predictions, labels, num_classes):
    """Calculates the categorical Dice similarity coefficients for each class
        between labels and predictions.

    Args:
        predictions (np.ndarray): predictions
        labels (np.ndarray): labels
        num_classes (int): number of classes to calculate the dice
            coefficient for

    Returns:
        np.ndarray: dice coefficient per class or NaN if class not present
    """

    dice_scores = np.zeros((num_classes))
    for i in range(num_classes):
        tmp_den = (np.sum(predictions == i) + np.sum(labels == i))
        tmp_dice = 2. * np.sum((predictions == i) * (labels == i)) / tmp_den
        dice_scores[i] = tmp_dice
    return dice_scores.astype(np.float32)


def abs_vol_difference(predictions, labels, num_classes):
    """Calculates the absolute volume difference for each class between
        labels and predictions.

    Args:
        predictions (np.ndarray): predictions
        labels (np.ndarray): labels
        num_classes (int): number of classes to calculate avd for

    Returns:
        np.ndarray: avd per class
    """

    avd = np.zeros((num_classes))
    eps = 1e-6
    for i in range(num_classes):
        avd[i] = np.abs(np.sum(predictions == i) - np.sum(labels == i)
                        ) / (np.float(np.sum(labels == i)) + eps)

    return avd.astype(np.float32)


def crossentropy(predictions, labels, logits=True):
    """Calculates the crossentropy loss between predictions and labels

    Args:
        prediction (np.ndarray): predictions
        labels (np.ndarray): labels
        logits (bool): flag whether predictions are logits or probabilities

    Returns:
        float: crossentropy error
    """

    if logits:
        maxes = np.amax(predictions, axis=-1, keepdims=True)
        softexp = np.exp(predictions - maxes)
        softm = softexp / np.sum(softexp, axis=-1, keepdims=True)
    else:
        softm = predictions
    loss = np.mean(-1. * np.sum(labels * np.log(softm + 1e-8), axis=-1))
    return loss.astype(np.float32)
