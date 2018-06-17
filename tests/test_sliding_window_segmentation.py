import tensorflow as tf
from dltk.utils import sliding_window_segmentation_inference
import numpy as np


def test_sw_inference():

    inp = tf.placeholder(tf.float32, [1, 1, 2, 1])
    op = tf.ones([1, 1, 2, 1], tf.float32)
    np_inp = np.ones([1, 4, 4, 1])

    with tf.Session() as s:
        out = sliding_window_segmentation_inference(s, [op], {inp: np_inp})[0]
        assert np.isclose(out, np_inp).all(), \
            'Got {} but expected {}'.format(out, np_inp)
