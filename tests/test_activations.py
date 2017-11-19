import tensorflow as tf
from dltk.core.activations import leaky_relu
import numpy as np


def test_leaky_relu():
    test_alpha = tf.constant(0.1)
    test_inp_1 = tf.constant(1.)
    test_inp_2 = tf.constant(-1.)

    test_relu_1 = leaky_relu(test_inp_1, test_alpha)
    test_relu_2 = leaky_relu(test_inp_2, test_alpha)

    with tf.Session() as s:
        out_1 = s.run(test_relu_1)
        assert np.isclose(out_1, 1.), \
            'Got {} but expected {}'.format(out_1, 1.)

        out_2 = s.run(test_relu_2)
        assert np.isclose(out_2, -0.1), \
            'Got {} but expected {}'.format(out_2, -0.1)
