import numpy as np


def dice(pred, labels, num_classes):
    """Calculates the dice score of labels and predictions

    Parameters
    ----------
    pred : np.ndarray
        predictions
    labels : np.ndarray
        labels
    num_classes : int
        number of classes to calculate avd for

    Returns
    -------
    np.ndarray
        dice per class

    """

    dice_scores = np.zeros((num_classes))
    for i in range(num_classes):
        tmp_den = (np.sum(pred == i) + np.sum(labels == i))
        tmp_dice = 2. * np.sum((pred == i) * (labels == i)) / tmp_den if tmp_den > 0 else 1.
        dice_scores[i] = tmp_dice
    return dice_scores


def abs_vol_difference(pred, labels, num_classes):
    """Calculates the average volume difference of labels and predictions per class

    Parameters
    ----------
    pred : np.ndarray
        predictions
    labels : np.ndarray
        labels
    num_classes : int
        number of classes to calculate avd for

    Returns
    -------
    np.ndarray
        avd per class

    """

    avd = np.zeros((num_classes))
    eps = 1e-6
    for i in range(num_classes):
        avd[i] = np.abs(np.sum(pred == i) - np.sum(labels == i)) / (np.float(np.sum(labels == i)) + eps)
        
    return avd


def crossentropy(pred, labels, logits=True):
    """ Calculates the crossentropy loss between prediction and labels

    Parameters
    ----------
    pred : np.ndarray
        prediction of the system
    labels : np.ndarray
        labels
    logits : bool
        flag whether pred are logits or probabilities

    Returns
    -------
    float
        crossentropy error

    """
    if logits:
        maxes = np.amax(pred, axis=-1, keepdims=True)
        softexp = np.exp(pred - maxes)
        softm = softexp / np.sum(softexp, axis=-1, keepdims=True)
    else:
        softm = pred
    loss = np.mean(-1. * np.sum(labels * np.log(softm), axis=-1))
    return loss