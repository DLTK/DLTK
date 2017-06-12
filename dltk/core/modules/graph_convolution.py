from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import numpy as np
import scipy

from dltk.core.modules.base import AbstractModule


class GraphConvolution(AbstractModule):
    """Graph Convolution module using Chebyshev polynomials

    This module builds a graph convolution using the Chebyshev polynomial filters proposed by Defferrard et al. (2016).

    """
    def __init__(self, out_filters, laplacian, k=3, bias='b1', name='gconv'):
        """Constructs the graph convolution template

        Parameters
        ----------
        out_filters : int
            number of output filters
        laplacian : numpy array
            graph Laplacian to use as a basis for the filters
        k : int, optional
            order of the Chebyshev polynomial
        bias : string
            type of bias to use, 'b1' for one bias per filter or 'b2' for one bias per vertex per filter
        name : string
            name of the module
        """

        assert (bias == 'b1' or bias == 'b2', 'Bias type must be either b1 or b2')

        self.in_filters = None
        self.out_filters = out_filters
        self.L = laplacian
        self.K = k
        self.bias = bias

        super(GraphConvolution, self).__init__(name=name)

    def rescale_L(self, L, lmax=2):
        """Rescale the Laplacian eigenvalues in [-1,1]."""
        M, M = L.shape
        I = scipy.sparse.identity(M, format='csr', dtype=L.dtype)
        L /= lmax / 2
        L -= I
        return L

    def _build(self, inp):
        """Applies a graph convolution operation to an input tensor

        Parameters
        ----------
        inp : tf.Tensor
            input tensor to be convolved

        Returns
        -------
        tf.Tensor
            convolved tensor

        """
        assert (len(inp.get_shape().as_list()) == 3, 'Graph Convolutional Layer needs 3D input.')

        self.in_shape = tuple(inp.get_shape().as_list())
        if self.in_filters is None:
            self.in_filters = self.in_shape[-1]
        assert(self.in_filters == self.in_shape[-1], 'Convolution was built for different number of input filters')

        N, M, self.in_filters = inp.get_shape()
        N, M, Fin = int(N), int(M), int(self.in_filters)
        # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.
        L = scipy.sparse.csr_matrix(self.L)
        L = self.rescale_L(L, lmax=2)
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        L = tf.sparse_reorder(L)
        # Transform to Chebyshev basis
        x0 = tf.transpose(inp, perm=[1, 2, 0])  # M x Fin x N
        x0 = tf.reshape(x0, [M, Fin * N])  # M x Fin*N
        x = tf.expand_dims(x0, 0)  # 1 x M x Fin*N

        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
            return tf.concat([x, x_], 0)  # K x M x Fin*N

        # recursive computation of the filters
        if self.K > 1:
            x1 = tf.sparse_tensor_dense_matmul(L, x0)
            x = concat(x, x1)
        for k in range(2, self.K):
            x2 = 2 * tf.sparse_tensor_dense_matmul(L, x1) - x0  # M x Fin*N
            x = concat(x, x2)
            x0, x1 = x1, x2
        x = tf.reshape(x, [self.K, M, Fin, N])  # K x M x Fin x N
        x = tf.transpose(x, perm=[3, 1, 2, 0])  # N x M x Fin x K
        x = tf.reshape(x, [N * M, Fin * self.K])  # N*M x Fin*K

        # Filter: Fin*out_filters filters of order K, i.e. one filterbank per feature pair.
        w_shape = [Fin * self.K, self.out_filters]
        initial = tf.truncated_normal_initializer(0, 0.1)
        self._w = tf.get_variable('w', shape=w_shape, dtype=tf.float32, initializer=initial,
                                  collections=self.WEIGHT_COLLECTIONS)
        self.variables.append(self._w)

        x = tf.matmul(x, self._w)  # N*M x out_filters
        x = tf.reshape(x, [N, M, self.out_filters])  # N x M x out_filters

        if self.bias == 'b1':
            b_shape = [1, 1, self.out_filters]
        elif self.bias == 'b2':
            b_shape = [1, M, self.out_filters]

        self._b = tf.get_variable("b", shape=b_shape, initializer=tf.constant_initializer(),
                                  collections=self.BIAS_COLLECTIONS)
        outp = x + self._b

        return outp
