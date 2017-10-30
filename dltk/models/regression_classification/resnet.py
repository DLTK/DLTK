"""Summary
"""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import numpy as np

from dltk.core.residual_unit import *


def resnet_3D(inputs, num_classes, num_res_units=1, filters=(16, 32, 64, 128), 
                    strides=((1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2)), 
                    mode=tf.estimator.ModeKeys.EVAL, name='resnet_3D'):
    """Regression/classification network based on a resnet architecture [1]. 

    [1] K. He et al. Deep residual learning for image recognition. CVPR 2016.
    
    Args:
        inputs (TYPE): Description
        num_classes (TYPE): Description
        num_res_units (int, optional): Description
        filters (tuple, optional): Description
        strides (tuple, optional): Description
        mode (TYPE, optional): Description
        name (str, optional): Description
    
    Returns:
        TYPE: Description
    """
    outputs = {}
    assert len(strides) == len(filters)
    assert len(inputs.get_shape().as_list()) == 5, 'inputs are required to have a rank of 5.'

    relu_op = tf.nn.relu6
    
    conv_params = {'padding' : 'same',
                  'use_bias' : False,
                  'kernel_initializer' : tf.uniform_unit_scaling_initializer(),
                  'bias_initializer' : tf.zeros_initializer(),
                  'kernel_regularizer' : None,
                  'bias_regularizer' : None}
    
    x = inputs
    
    # Inital convolution with filters[0]
    k = [s * 2 if s > 1 else 3 for s in strides[0]]
    x = tf.layers.conv3d(x, filters[0], k, strides[0], **conv_params)
    tf.logging.info('Init conv tensor shape {}'.format(x.get_shape()))

    # Residual feature encoding blocks with num_res_units at different resolution scales res_scales
    res_scales = [x]
    saved_strides = []
    for res_scale in range(1, len(filters)):
        
        # Features are downsampled via strided convolutions. These are defined in `strides` and subsequently saved
        with tf.variable_scope('unit_{}_0'.format(res_scale)):
            x = vanilla_residual_unit_3D(x, filters[res_scale], strides=strides[res_scale], mode=mode)
        saved_strides.append(strides[res_scale])
        
        for i in range(1, num_res_units):
            with tf.variable_scope('unit_{}_{}'.format(res_scale, i)):
                x = vanilla_residual_unit_3D(x, filters[res_scale], strides=(1, 1, 1), mode=mode)
        res_scales.append(x)
        tf.logging.info('Encoder at res_scale {} tensor shape: {}'.format(res_scale, x.get_shape()))
    
    # Global pool and last unit
    with tf.variable_scope('pool'):
        x = tf.layers.batch_normalization(x, training=mode==tf.estimator.ModeKeys.TRAIN)
        x = relu_op(x)  
        axis = tuple(range(len(x.get_shape().as_list())))[1:-1]
        x = tf.reduce_mean(x, axis=axis, name='global_avg_pool')
        tf.logging.info('Global pool shape {}'.format(x.get_shape()))
        
    with tf.variable_scope('last'):    
        x = tf.layers.dense(x,
                            num_classes,
                            activation=None,
                            use_bias=conv_params['use_bias'],
                            kernel_initializer=conv_params['kernel_initializer'],
                            bias_initializer=conv_params['bias_initializer'],
                            kernel_regularizer=conv_params['kernel_regularizer'],
                            bias_regularizer=conv_params['bias_regularizer'],
                            name='hidden_units')
        
        tf.logging.info('Output tensor shape {}'.format(x.get_shape()))

    # Define the outputs
    outputs['logits'] = x
    
    with tf.variable_scope('pred'):
        y_prob = tf.nn.softmax(x)
        outputs['y_prob'] = y_prob
        y_ = tf.argmax(x, axis=-1) if num_classes > 1 else tf.cast(tf.greater_equal(x[..., 0], 0.5), tf.int32)
        outputs['y_'] = y_

    return outputs