"""Summary
"""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import numpy as np

def vanilla_residual_unit_3D(inputs, out_filters, in_filters=None, kernel_size=(3, 3, 3), strides=(1, 1, 1), mode=tf.estimator.ModeKeys.EVAL, name='res_unit'):
    """Summary
    
    Args:
        inputs (TYPE): Description
        out_filters (TYPE): Description
        in_filters (None, optional): Description
        kernel_size (tuple, optional): Description
        strides (tuple, optional): Description
        mode (TYPE, optional): Description
        name (str, optional): Description
    
    Returns:
        TYPE: Description
    """

    relu_op = tf.nn.relu6 #or tf.nn.relu
    pool_op = tf.layers.max_pooling3d
    
    conv_params = {'padding': 'same',
                  'use_bias' : False,
                  'kernel_initializer' : tf.uniform_unit_scaling_initializer(),
                  'bias_initializer' : tf.zeros_initializer(),
                  'kernel_regularizer' : None,
                  'bias_regularizer' : None}
    
    if in_filters is None:
        in_filters = inputs.get_shape().as_list()[-1]
    assert in_filters == inputs.get_shape().as_list()[-1], 'Module was initialised for a different input shape'
        
    x = inputs
    orig_x = x
    
    # Handle strided convolutions
    if np.prod(strides) != 1:
        kernel_size = strides
        orig_x = pool_op(orig_x, strides, strides, 'valid')
    
    # Sub unit 0
    with tf.variable_scope('sub_unit0'):
        x = tf.layers.batch_normalization(x, training=mode==tf.estimator.ModeKeys.TRAIN)
        x = relu_op(x)
        x = tf.layers.conv3d(x, out_filters, kernel_size, strides, **conv_params)
        
    # Sub unit 1
    with tf.variable_scope('sub_unit1'):
        x = tf.layers.batch_normalization(x, training=mode==tf.estimator.ModeKeys.TRAIN)
        x = relu_op(x)
        x = tf.layers.conv3d(x, out_filters, kernel_size, (1, 1, 1), **conv_params)

    # Add the residual
    with tf.variable_scope('sub_unit_add'):

        # Handle differences in input and output filter sizes
        if in_filters < out_filters:
            orig_x = tf.pad(orig_x, [[0, 0]] * (len(x.get_shape().as_list()) - 1) +
                                     [[int(np.floor((out_filters - in_filters) / 2.)),
                                      int(np.ceil((out_filters - in_filters) / 2.))]])
        elif in_filters > out_filters:
            orig_x = tf.layers.conv3d(orig_x, out_filters, kernel_size, (1, 1, 1), **conv_params)
        x += orig_x
        
    return x