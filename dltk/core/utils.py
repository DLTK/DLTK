import numpy as np
from dltk.core.io.sliding_window import SlidingWindow

def sliding_window_segmentation_inference(session, ops_list, sample_dict, batch_size=1):
    """

    Parameters
    ----------
    session
    ops_list
    sample_dict

    Returns
    -------

    """

    # TODO: asserts

    pl_shape = list(sample_dict.keys()[0].get_shape().as_list())

    pl_bshape = pl_shape[1:-1]

    inp_shape = list(sample_dict.values()[0].shape)
    inp_bshape = inp_shape[1:-1]

    out_dummies = [np.zeros([inp_shape[0], ] + inp_bshape + [op.get_shape().as_list()[-1]]
                            if len(op.get_shape().as_list()) == len(inp_shape) else []) for op in ops_list]

    out_dummy_counter = [np.zeros_like(o) for o in out_dummies]

    op_shape = list(ops_list[0].get_shape().as_list())
    op_bshape = op_shape[1:-1]

    out_diff = np.array(pl_bshape) - np.array(op_bshape)

    padding = [[0, 0]] + [[diff // 2, diff - diff // 2] for diff in out_diff] + [[0, 0]]

    padded_dict = {k: np.pad(v, padding, mode='constant') for k,v in sample_dict.items()}

    f_bshape = padded_dict.values()[0].shape[1:-1]

    striding = list(np.array(op_bshape) // 2) if all(out_diff == 0) else op_bshape

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
            sw_dict = {k: v[slicer] for k,v in padded_dict.items()}
            op_parts = session.run(ops_list, feed_dict=sw_dict)

            for idx in range(len(op_parts)):
                out_dummies[idx][out_slicer] += op_parts[idx]
                out_dummy_counter[idx][out_slicer] += 1
        else:
            slicers.append(slicer)
            out_slicers.append(out_slicer)
            if len(slicers) == batch_size or done:
                slices_dict = {k: np.concatenate([v[slicer] for slicer in slicers], 0) for k,v in padded_dict.items()}

                all_op_parts = session.run(ops_list, feed_dict=slices_dict)

                zipped_parts = zip(*[np.array_split(part, len(slicers)) for part in all_op_parts])

                for out_slicer, op_parts in zip(out_slicers, zipped_parts):
                    for idx in range(len(op_parts)):
                        out_dummies[idx][out_slicer] += op_parts[idx]
                        out_dummy_counter[idx][out_slicer] += 1

                slicers = []
                out_slicers = []

        if done:
            break

    return [o / c for o, c in zip(out_dummies, out_dummy_counter)]
