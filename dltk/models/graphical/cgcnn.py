from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import numpy as np

from dltk.core.modules.base import AbstractModule
from dltk.core.modules.linear import Linear
from dltk.core.modules.graph_convolution import GraphConvolution


class CGCNN(AbstractModule):
    """Graph CNN

    This module builds a Graph Convolutional Network using the Chebyshev approximation (Defferrard et al. 2016).
    """
    def __init__(self, L, filters, K, p, fc, bias='b1', pool='mpool', dropout=1, name='cgcnn'):

        """Builds a graph CNN for classification

        Parameters
        ----------
        L: list
            list of Graph Laplacians, one per coarsening level
        filters : list
            number of filters per layer
        K : list
            list of polynomial orders, i.e. filter sizes or number of hopes
        p : list
            pooling size (should be 1 - no pooling or a power of 2 - reduction by 2 at each coarser level)
        fc: list
            number of features per sample, i.e. number of hidden neurons
            (the last layer is the softmax, i.e. M[-1] is the number of classes)
        bias: string
            type of bias to use, 'b1' for one bias per filter or 'b2' for one bias per vertex per filter
        pool: string
            pooling, 'mpool' for max pooling or 'apool' for average pooling
        dropout: float
            dropout for fc layers, probability to keep hidden neurons (no dropout with 1)

        name : string
            name of the network
        """

        # Verify the consistency w.r.t. the number of layers
        assert len(L) >= len(filters) == len(K) == len(p)
        assert np.all(np.array(p) >= 1)
        p_log2 = np.where(np.array(p) > 1, np.log2(p), 0)
        assert np.all(np.mod(p_log2, 1) == 0)  # Powers of 2.
        assert len(L) >= 1 + np.sum(p_log2)  # Enough coarsening levels for pool sizes.

        # Keep the useful Laplacians only
        M_0 = L[0].shape[0]
        j = 0
        self.L = []
        for pp in p:
            self.L.append(L[j])
            j += int(np.log2(pp)) if pp > 1 else 0
        L = self.L

        # Store attributes
        self.L = L
        self.filters = filters
        self.K = K
        self.p = p
        self.fc = fc
        self.bias = bias
        self.pool = pool
        self.dropout = dropout

        super(CGCNN, self).__init__(name)

    def _build(self, inp, is_training=True):
        """Constructs a GraphCNN using the input tensor

        Parameters
        ----------
        inp : tf.Tensor
            input tensor
        is_training : bool
            flag to specify whether this is training - used for dropout

        Returns
        -------
        dict
            output dictionary containing:
                - `logits` - logits of the classification
                - `y_prob` - classification probabilities
                - `y_` - prediction of the classification

        """
        outputs = {}

        x = inp
        pool_op = tf.nn.max_pool if self.pool == 'mpool' else tf.nn.avg_pool

        # Graph convolutional layers
        x = tf.expand_dims(x, 2)  # N x M x F=1
        for i in range(len(self.p)):
            with tf.variable_scope('conv_{}'.format(i + 1)):
                with tf.name_scope('filter'):
                    x = GraphConvolution(self.filters[i], self.L[i], self.K[i], self.bias)(x)
                with tf.name_scope('relu'):
                    x = tf.nn.relu(x)
                with tf.name_scope('pooling'):
                    if self.p[i] > 1:
                        x = tf.expand_dims(x, 3)  # N x M x F x 1
                        x = pool_op(x, ksize=[1, self.p[i], 1, 1], strides=[1, self.p[i], 1, 1],
                                    padding='SAME')
                        x = tf.squeeze(x, [3])  # N x M/p x F

        # Fully connected hidden layers
        N, M, F = x.get_shape()
        x = tf.reshape(x, [int(N), int(M * F)])  # N x M
        for i, M in enumerate(self.fc[:-1]):
            with tf.variable_scope('fc_{}'.format(i + 1)):
                x = Linear(M)(x)
                x = tf.nn.relu(x)
                if is_training:
                    x = tf.nn.dropout(x, self.dropout)

        # Logits linear layer, i.e. softmax without normalization.
        with tf.variable_scope('logits'):
            x = Linear(self.fc[-1])(x)

        outputs['logits'] = x

        with tf.variable_scope('pred'):
            y_prob = tf.nn.softmax(x)
            outputs['y_prob'] = y_prob
            y_ = tf.argmax(x, axis=1)
            outputs['y_'] = y_

        return outputs