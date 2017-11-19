from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
import numpy as np


def get_linear_upsampling_kernel(kernel_spatial_shape,
                                 out_filters,
                                 in_filters,
                                 trainable=False):
    """Builds a kernel for linear upsampling with the shape
        [kernel_spatial_shape] + [out_filters, in_filters]. Can be set to
        trainable to potentially learn a better upsamling.

    Args:
        kernel_spatial_shape (list or tuple): Spatial dimensions of the
            upsampling kernel. Is required to be of rank 2 or 3,
            (i.e. [dim_x, dim_y] or [dim_x, dim_y, dim_z])
        out_filters (int): Number of output filters.
        in_filters (int): Number of input filters.
        trainable (bool, optional): Flag to set the returned tf.Variable
            to be trainable or not.

    Returns:
        tf.Variable: Linear upsampling kernel
    """

    rank = len(list(kernel_spatial_shape))
    assert 1 < rank < 4, \
        'Transposed convolutions are only supported in 2D and 3D'

    kernel_shape = tuple(kernel_spatial_shape + [out_filters, in_filters])
    size = kernel_spatial_shape
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

    return tf.get_variable(name="linear_up_kernel",
                           initializer=init,
                           shape=weights.shape,
                           trainable=trainable)


def linear_upsample_3d(inputs,
                       strides=(2, 2, 2),
                       use_bias=False,
                       trainable=False,
                       name='linear_upsample_3d'):
    """Linear upsampling layer in 3D using strided transpose convolutions. The
        upsampling kernel size will be automatically computed to avoid
        information loss.

    Args:
        inputs (tf.Tensor): Input tensor to be upsampled
        strides (tuple, optional): The strides determine the upsampling factor
            in each dimension.
        use_bias (bool, optional): Flag to train an additional bias.
        trainable (bool, optional): Flag to set the variables to be trainable or not.
        name (str, optional): Name of the layer.

    Returns:
        tf.Tensor: Upsampled Tensor
    """
    static_inp_shape = tuple(inputs.get_shape().as_list())
    dyn_inp_shape = tf.shape(inputs)
    rank = len(static_inp_shape)

    num_filters = static_inp_shape[-1]
    strides_5d = [1, ] + list(strides) + [1, ]
    kernel_size = [2 * s if s > 1 else 1 for s in strides]

    kernel = get_linear_upsampling_kernel(
        kernel_spatial_shape=kernel_size,
        out_filters=num_filters,
        in_filters=num_filters,
        trainable=trainable)

    dyn_out_shape = [dyn_inp_shape[i] * strides_5d[i] for i in range(rank)]
    dyn_out_shape[-1] = num_filters

    static_out_shape = [static_inp_shape[i] * strides_5d[i]
                        if isinstance(static_inp_shape[i], int)
                        else None for i in range(rank)]

    static_out_shape[-1] = num_filters
    tf.logging.info('Upsampling from {} to {}'.format(
        static_inp_shape, static_out_shape))

    upsampled = tf.nn.conv3d_transpose(
        value=inputs,
        filter=kernel,
        output_shape=dyn_out_shape,
        strides=strides_5d,
        padding='SAME',
        name='upsample')

    upsampled.set_shape(static_out_shape)

    return upsampled
