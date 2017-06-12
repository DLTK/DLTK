from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
from dltk.core.modules.summaries import *


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