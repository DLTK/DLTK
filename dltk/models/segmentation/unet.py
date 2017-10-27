"""Summary
"""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import numpy as np

from dltk.core.residual_unit import *
from dltk.core.upsample import *

def upsample_and_concat(inputs, inputs2, strides=(2, 2, 2), name='up_and_concat'):
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

    assert len(inputs.get_shape().as_list()) == len(inputs2.get_shape().as_list()), 'Ranks of input and input2 differ'
        
    # Upsample inputs
    inputs = linear_upsample_3D(inputs, strides) 
    
    return tf.concat(axis=-1,values=[inputs2, inputs])


def residual_unet_3D(inputs, num_classes, num_res_units=1, filters=(16, 32, 64, 128), 
                    strides=((1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2)), 
                    mode=tf.estimator.ModeKeys.EVAL, name='residual_fcn_3D'):
    """Image segmentation network based on an UNET architecture [1] using residual units [2] as feature extractors. 

    [1] O. Ronneberger et al. U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI 2015.
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
        with tf.variable_scope('enc_unit_{}_0'.format(res_scale)):
            x = vanilla_residual_unit_3D(x, filters[res_scale], strides=strides[res_scale], mode=mode)
        saved_strides.append(strides[res_scale])
        
        for i in range(1, num_res_units):
            with tf.variable_scope('enc_unit_{}_{}'.format(res_scale, i)):
                x = vanilla_residual_unit_3D(x, filters[res_scale], strides=(1, 1, 1), mode=mode)
        res_scales.append(x)
        tf.logging.info('Encoder at res_scale {} tensor shape: {}'.format(res_scale, x.get_shape()))

    # Upsample and concat layers [1] reconstruct the predictions to higher resolution scales
    for res_scale in range(len(filters) - 2, -1, -1):
        with tf.variable_scope('up_concat_{}'.format(res_scale)):
            x = upsample_and_concat(x, res_scales[res_scale], strides=saved_strides[res_scale])
            
        for i in range(0, num_res_units):
            with tf.variable_scope('dec_unit_{}_{}'.format(res_scale, i)):
                x = vanilla_residual_unit_3D(x, filters[res_scale], strides=(1, 1, 1), mode=mode)
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