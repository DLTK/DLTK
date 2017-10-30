"""Summary
"""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import numpy as np


def convolutional_autoencoder_3D(inputs, num_convolutions=1, num_hidden_units=128, filters=(16, 32, 64), 
                    strides=((2, 2, 2), (2, 2, 2), (2, 2, 2)), 
                    mode=tf.estimator.ModeKeys.EVAL, use_bias=False, name='conv_autoencoder_3D'):
    """Convolutional autoencoder with num_convolutions on len(filters) resolution scales. Downsampling features is done via strided convolutions. 
    
    Args:
        inputs (TYPE): Description
        num_convolutions (int, optional): Description
        num_hidden_units(int, optional): Description
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

    conv_op = tf.layers.conv3d
    tp_conv_op = tf.layers.conv3d_transpose
    relu_op = tf.nn.relu6
    
    conv_params = {'padding' : 'same',
                  'use_bias' : use_bias,
                  'kernel_initializer' : tf.uniform_unit_scaling_initializer(),
                  'bias_initializer' : tf.zeros_initializer(),
                  'kernel_regularizer' : None,
                  'bias_regularizer' : None}

    x = inputs
    tf.logging.info('Input tensor shape {}'.format(x.get_shape()))

    # Convolutional feature encoding blocks with num_convolutions at different resolution scales res_scales
    for res_scale in range(0, len(filters)):
        
        for i in range(0, num_convolutions - 1):
            with tf.variable_scope('enc_unit_{}_{}'.format(res_scale, i)):
                x = conv_op(x, filters[res_scale], (3, 3, 3), (1, 1, 1), **conv_params)
                x = tf.layers.batch_normalization(x, training=mode==tf.estimator.ModeKeys.TRAIN)
                x = relu_op(x)       
                tf.logging.info('Encoder at res_scale {} tensor shape: {}'.format(res_scale, x.get_shape()))
        
        # Employ strided convolutions to downsample
        with tf.variable_scope('enc_unit_{}_{}'.format(res_scale, num_convolutions)):
            
            # Adjust the strided conv kernel size to prevent losing information
            k_size = [s * 2 if s > 1 else 3 for s in strides[res_scale]]
            
            x = conv_op(x, filters[res_scale], k_size, strides[res_scale], **conv_params)
            x = tf.layers.batch_normalization(x, training=mode==tf.estimator.ModeKeys.TRAIN)
            x = relu_op(x)  
            tf.logging.info('Encoder at res_scale {} tensor shape: {}'.format(res_scale, x.get_shape()))
        
    # Densely connected layer of hidden units 
    x_shape = x.get_shape().as_list() 
    x = tf.reshape(x, (tf.shape(x)[0], np.prod(x_shape[1:])))        
    
    x = tf.layers.dense(x,
                        num_hidden_units,
                        use_bias=conv_params['use_bias'],
                        kernel_initializer=conv_params['kernel_initializer'],
                        bias_initializer=conv_params['bias_initializer'],
                        kernel_regularizer=conv_params['kernel_regularizer'],
                        bias_regularizer=conv_params['bias_regularizer'],
                        name='hidden_units')

    outputs['hidden_units'] = x
    tf.logging.info('Hidden units tensor shape: {}'.format(x.get_shape()))
    
    x = tf.layers.dense(x,
                        np.prod(x_shape[1:]),
                        activation=relu_op,
                        use_bias=conv_params['use_bias'],
                        kernel_initializer=conv_params['kernel_initializer'],
                        bias_initializer=conv_params['bias_initializer'],
                        kernel_regularizer=conv_params['kernel_regularizer'],
                        bias_regularizer=conv_params['bias_regularizer'])

    x = tf.reshape(x, [tf.shape(x)[0]] + list(x_shape)[1:])
    tf.logging.info('Decoder input tensor shape: {}'.format(x.get_shape()))
    
    # Decoding blocks with num_convolutions at different resolution scales res_scales
    for res_scale in reversed(range(0, len(filters))):
        
        # Employ strided transpose convolutions to upsample
        with tf.variable_scope('dec_unit_{}_0'.format(res_scale)):
            
            # Adjust the strided tp conv kernel size to prevent losing information
            k_size = [s * 2 if s > 1 else 3 for s in strides[res_scale]]
            
            x = tp_conv_op(x, filters[res_scale], k_size, strides[res_scale], **conv_params)
            x = tf.layers.batch_normalization(x, training=mode==tf.estimator.ModeKeys.TRAIN)
            x = relu_op(x)
            tf.logging.info('Decoder at res_scale {} tensor shape: {}'.format(res_scale, x.get_shape()))

        for i in range(1, num_convolutions):
            with tf.variable_scope('dec_unit_{}_{}'.format(res_scale, i)):
                x = conv_op(x, filters[res_scale], (3, 3, 3), (1, 1, 1), **conv_params)
                x = tf.layers.batch_normalization(x, training=mode==tf.estimator.ModeKeys.TRAIN)
                x = relu_op(x)  
            tf.logging.info('Decoder at res_scale {} tensor shape: {}'.format(res_scale, x.get_shape())) 
     
    # A final convolution reduces the number of output features to those of the inputs
    x = conv_op(x, inputs.get_shape().as_list()[-1], (1, 1, 1), (1, 1, 1), **conv_params)
    
    tf.logging.info('Output tensor shape: {}'.format(x.get_shape()))
    outputs['x_'] = x
    
    return outputs