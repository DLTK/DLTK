from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np


class SlidingWindow(object):
    """SlidingWindow
    Sliding window iterator which produces slice objects to slice in a sliding
    window. This is useful for inference.
    """

    def __init__(self, img_shape, window_shape, has_batch_dim=True,
                 striding=None):
        """Constructs a sliding window iterator

        Args:
            img_shape (array_like): shape of the image to slide over

            window_shape (array_like): shape of the window to extract

            has_batch_dim (bool, optional): flag to indicate whether a batch
                dimension is present

            striding (array_like, optional): amount to move the window between
                each position
        """

        self.img_shape = img_shape
        self.window_shape = window_shape
        self.rank = len(img_shape)
        self.curr_pos = [0] * self.rank
        self.end_pos = [0] * self.rank
        self.done = False
        self.striding = window_shape
        self.has_batch_dim = has_batch_dim
        if striding:
            self.striding = striding

    def __iter__(self):
        return self

    # py 2.* compatability hack
    def next(self):
        return self.__next__()

    def __next__(self):
        if self.done:
            raise StopIteration()

        if self.has_batch_dim:
            slicer = [slice(None)] * (self.rank + 1)
        else:
            slicer = [slice(None)] * self.rank

        move_dim = True
        for dim, pos in enumerate(self.curr_pos):
            low = pos
            high = pos + self.window_shape[dim]
            if move_dim:
                if high >= self.img_shape[dim]:
                    self.curr_pos[dim] = 0
                    move_dim = True
                else:
                    self.curr_pos[dim] += self.striding[dim]
                    move_dim = False
            if high >= self.img_shape[dim]:
                low = self.img_shape[dim] - self.window_shape[dim]
                high = self.img_shape[dim]

            if self.has_batch_dim:
                slicer[dim + 1] = slice(low, high)
            else:
                slicer[dim] = slice(low, high)

        if (np.array(self.curr_pos) == self.end_pos).all():
            self.done = True
        return slicer


def sliding_window_segmentation_inference(session,
                                          ops_list,
                                          sample_dict,
                                          batch_size=1,
                                          striding=None):
    """
    Utility function to perform sliding window inference for segmentation

    Args:
        session (tf.Session): TensorFlow session to run ops with

        ops_list (array_like): Operators to fetch assemble with sliding window

        sample_dict (dict): Dictionary with tf.Placeholder keys mapping the
        placeholders to their respective input

        batch_size (int, optional): Number of sliding windows to batch for
            calculation

        striding (array_like): Striding of the sliding window. Can be used to
            adjust overlap etc.

    Returns:
        list: List of np.arrays corresponding to the assembled outputs of
            ops_list
    """

    # TODO: asserts
    assert batch_size > 0, 'Batch size has to be 1 or bigger'

    pl_shape = list(sample_dict.keys())[0].get_shape().as_list()

    pl_bshape = pl_shape[1:-1]

    inp_shape = list(list(sample_dict.values())[0].shape)
    inp_bshape = inp_shape[1:-1]

    out_dummies = [np.zeros(
        [inp_shape[0], ] + inp_bshape + [op.get_shape().as_list()[-1]]
        if len(op.get_shape().as_list()) == len(inp_shape) else []) for op in ops_list]

    out_dummy_counter = [np.zeros_like(o) for o in out_dummies]

    op_shape = list(ops_list[0].get_shape().as_list())
    op_bshape = op_shape[1:-1]

    out_diff = np.array(pl_bshape) - np.array(op_bshape)

    padding = [[0, 0]] + [[diff // 2, diff - diff // 2] for diff
                          in out_diff] + [[0, 0]]

    padded_dict = {k: np.pad(v, padding, mode='constant') for k, v
                   in sample_dict.items()}

    f_bshape = list(padded_dict.values())[0].shape[1:-1]

    if not striding:
        striding = (list(np.maximum(1, np.array(op_bshape) // 2))
                    if all(out_diff == 0) else op_bshape)

    sw = SlidingWindow(f_bshape, pl_bshape, striding=striding)
    out_sw = SlidingWindow(inp_bshape, op_bshape, striding=striding)

    if batch_size > 1:
        slicers = []
        out_slicers = []

    done = False
    while True:
        try:
            slicer = next(sw)
            out_slicer = next(out_sw)
        except StopIteration:
            done = True

        if batch_size == 1:
            sw_dict = {k: v[slicer] for k, v in padded_dict.items()}
            op_parts = session.run(ops_list, feed_dict=sw_dict)

            for idx in range(len(op_parts)):
                out_dummies[idx][out_slicer] += op_parts[idx]
                out_dummy_counter[idx][out_slicer] += 1
        else:
            slicers.append(slicer)
            out_slicers.append(out_slicer)
            if len(slicers) == batch_size or done:
                slices_dict = {k: np.concatenate(
                    [v[slicer] for slicer in slicers], 0) for k, v in padded_dict.items()}

                all_op_parts = session.run(ops_list, feed_dict=slices_dict)

                zipped_parts = zip(*[np.array_split(part, len(slicers)) for
                                     part in all_op_parts])

                for out_slicer, op_parts in zip(out_slicers, zipped_parts):
                    for idx in range(len(op_parts)):
                        out_dummies[idx][out_slicer] += op_parts[idx]
                        out_dummy_counter[idx][out_slicer] += 1

                slicers = []
                out_slicers = []

        if done:
            break

    return [o / c for o, c in zip(out_dummies, out_dummy_counter)]
