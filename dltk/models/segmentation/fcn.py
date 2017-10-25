"""Summary
"""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import numpy as np

from dltk.core.common import *

def upscore_layer_3D(inputs, inputs2, out_filters, in_filters=None, strides=(2, 2, 2), mode=tf.estimator.ModeKeys.EVAL, name='upscore'):
    """Summary
    
    Args:
        inputs (TYPE): Description
        inputs2 (TYPE): Description
        out_filters (TYPE): Description
        in_filters (None, optional): Description
        strides (tuple, optional): Description
        mode (TYPE, optional): Description
        name (str, optional): Description
    
    Returns:
        TYPE: Description
    """
    conv_params = {'padding' : 'same',
                  'use_bias' : False,
                  'kernel_initializer' : tf.uniform_unit_scaling_initializer(),
                  'bias_initializer' : tf.zeros_initializer(),
                  'kernel_regularizer' : None,
                  'bias_regularizer' : None}

    # Compute an upsampling shape dynamically from the input tensor. Input filters are required to be static.
    if in_filters is None:
        in_filters = inputs.get_shape().as_list()[-1]
        
    assert in_filters == inputs.get_shape().as_list()[-1], 'Module was initialised for a different input shape'
    
    # Account for differences in the number of input and output filters
    if in_filters != out_filters:
        x = tf.layers.conv3d(inputs, out_filters, (1, 1, 1), (1, 1, 1), name='filter_conversion', **conv_params)
    else:
        x = inputs
    
    # Upsample inputs
    x = bilinear_upsample_3D(x, strides)    
        
    # Skip connection
    x2 = tf.layers.conv3d(inputs2, out_filters, (1, 1, 1), (1, 1, 1), **conv_params)
    x2 = tf.layers.batch_normalization(x2, training=mode==tf.estimator.ModeKeys.TRAIN)
    
    # Return the element-wise sum
    return tf.add(x, x2)


def residual_fcn_3D(inputs, num_classes, num_res_units=1, filters=(16, 32, 64, 128), 
                    strides=((1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2)), 
                    mode=tf.estimator.ModeKeys.EVAL, name='residual_fcn_3D'):
    """Image segmentation network based on an FCN architecture [1] using residual units [2] as feature extractors. 

    [1] J. Long et al. Fully convolutional networks for semantic segmentation. CVPR 2015.
    [2] K. He et al. Deep residual learning for image recognition. CVPR 2016.
    
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

    conv_params = {'padding' : 'same',
                  'use_bias' : False,
                  'kernel_initializer' : tf.uniform_unit_scaling_initializer(),
                  'bias_initializer' : tf.zeros_initializer(),
                  'kernel_regularizer' : None,
                  'bias_regularizer' : None}
    
    x = inputs
    
    # Inital convolution with filters[0]
    x = tf.layers.conv3d(x, filters[0], (3, 3, 3), strides[0], **conv_params)
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

    # Upscore layers [2] reconstruct the predictions to higher resolution scales
    for res_scale in range(len(filters) - 2, -1, -1):
        with tf.variable_scope('upscore_{}'.format(res_scale)):
            x = upscore_layer_3D(x, res_scales[res_scale], num_classes, strides=saved_strides[res_scale], mode=mode)
        tf.logging.info('Decoder at res_scale {} tensor shape: {}'.format(res_scale, x.get_shape()))

    # Last convolution
    with tf.variable_scope('last'):
        x = tf.layers.conv3d(x, num_classes, (1, 1, 1), (1, 1, 1), **conv_params)
    tf.logging.info('Output tensor shape {}'.format(x.get_shape()))

    # Define the outputs
    outputs['logits'] = x
    
    with tf.variable_scope('pred'):
        y_prob = tf.nn.softmax(x)
        outputs['y_prob'] = y_prob
        y_ = tf.argmax(x, axis=-1) if num_classes > 1 else tf.cast(tf.greater_equal(x[..., 0], 0.5), tf.int32)
        outputs['y_'] = y_

    return outputs