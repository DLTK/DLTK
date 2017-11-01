"""Summary
"""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import numpy as np

def get_linear_upsampling_kernel(up_spatial_shape, out_filters, in_filters, trainable=False):
    
        """builds kernel for linear upsampling
        
        Args:
            up_spatial_shape (TYPE): Description
            out_filters (TYPE): Description
            in_filters (TYPE): Description
            trainable (bool, optional): Description
        
        Returns:
            TYPE: Description
        """
        
        rank = len(list(up_spatial_shape))
        assert 1 < rank < 4, 'Transposed convolutions are only supported in 2D and 3D'
        
        kernel_shape = tuple(up_spatial_shape + [out_filters, in_filters])
        size = up_spatial_shape
        factor = (np.array(size) + 1) // 2
        center = np.zeros_like(factor, np.float)

        for i in range(len(factor)):
            if size[i] % 2 == 1:
                center[i] = factor[i] - 1
            else:
                center[i] = factor[i] - 0.5

        weights = np.zeros(kernel_shape)
        if rank == 2:
            og = np.ogrid[:size[0], :size[1]]
            x_filt = (1 - abs(og[0] - center[0]) / np.float(factor[0]))
            y_filt = (1 - abs(og[1] - center[1]) / np.float(factor[1]))

            filt = x_filt * y_filt

            for i in range(out_filters):
                weights[:, :, i, i] = filt
        else:
            og = np.ogrid[:size[0], :size[1], :size[2]]
            x_filt = (1 - abs(og[0] - center[0]) / np.float(factor[0]))
            y_filt = (1 - abs(og[1] - center[1]) / np.float(factor[1]))
            z_filt = (1 - abs(og[2] - center[2]) / np.float(factor[2]))

            filt = x_filt * y_filt * z_filt

            for i in range(out_filters):
                weights[:, :, :, i, i] = filt

        init = tf.constant_initializer(value=weights, dtype=tf.float32)
        
        return tf.get_variable(name="bilinear_up_kernel", initializer=init, shape=weights.shape, trainable=trainable)


def linear_upsample_3D(inputs, strides=(2, 2, 2), use_bias=False, trainable=False, name='linear_upsampling'):
    """Summary
    
    Args:
        inputs (TYPE): Description
        strides (tuple, optional): Description
        use_bias (bool, optional): Description
        trainable (bool, optional): Description
        name (str, optional): Description
    
    Returns:
        TYPE: Description
    """
    static_inp_shape = tuple(inputs.get_shape().as_list())
    dyn_inp_shape = tf.shape(inputs)
    rank = len(static_inp_shape)

    num_filters = static_inp_shape[-1]
    strides_5D = [1,] + list(strides) + [1,]
    kernel_size = [2 * s if s > 1 else 1 for s in strides]

    kernel = get_linear_upsampling_kernel(kernel_size, num_filters, num_filters, trainable)

    dyn_out_shape = [dyn_inp_shape[i] * strides_5D[i] for i in range(rank)]
    dyn_out_shape[-1] = num_filters

    static_out_shape = [static_inp_shape[i] * strides_5D[i] if isinstance(static_inp_shape[i], int) else None for i in range(rank)]
    static_out_shape[-1] = num_filters
    tf.logging.info('Upsampling from {} to {}'.format(static_inp_shape, static_out_shape))

    upsampled = tf.nn.conv3d_transpose(inputs, filter=kernel, output_shape=dyn_out_shape, strides=strides_5D, padding='SAME', name='upsample')
    upsampled.set_shape(static_out_shape)
    
    return upsampled    