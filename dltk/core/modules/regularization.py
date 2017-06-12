from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
from dltk.core.modules.summaries import *


def l2_regularization(variables, factor=1e-4, name='l2_regularization', collections=['regularization']):
    """ l2 regularization

    Calculates l2 penalty for given variables and constructs a scalar summary

    Parameters
    ----------
    variables : list or tuple
        list of variables to calculate the l2 penalty for
    factor : float
        factor to weight the penalty by
    name : string
        name of the summary
    collections : list or tuple
        collections to add the summary to

    Returns
    -------
    tf.Tensor
        l2 penalty for the variables given

    """
    l2 = tf.add_n([tf.sqrt(2.*tf.nn.l2_loss(var)) for var in variables], name=name) if variables else tf.constant(0.)
    loss = factor * l2
    scalar_summary(loss, name, collections)
    return loss


def l1_regularization(variables, factor=1e-4, name='l1_regularization', collections=['regularization']):
    """ l1 regularization

    Calculates l1 penalty for given variables and constructs a scalar summary

    Parameters
    ----------
    variables : list or tuple
        list of variables to calculate the l2 penalty for
    factor : float
        factor to weight the penalty by
    name : string
        name of the summary
    collections : list or tuple
        collections to add the summary to

    Returns
    -------
    tf.Tensor
        l2 penalty for the variables given

    """
    l1 = tf.add_n([tf.reduce_sum(tf.abs(var)) for var in variables], name=name) if variables else tf.constant(0.)
    loss = factor * l1
    scalar_summary(loss, name, collections)
    return loss