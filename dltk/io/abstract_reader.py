from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
import traceback


class IteratorInitializerHook(tf.train.SessionRunHook):
    """Hook to initialise data iterator after Session is created."""

    def __init__(self):
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_func = None

    def after_create_session(self, session, coord):
        """Initialise the iterator after the session has been created."""
        self.iterator_initializer_func(session)


class Reader(object):
    """Wrapper for dataset generation given a read function"""

    def __init__(self, read_fn, dtypes):
        """Constructs a Reader instance

        Args:
            read_fn: Input function returning features which is a dictionary of
                string feature name to `Tensor` or `SparseTensor`. If it
                returns a tuple, first item is extracted as features.
                Prediction continues until `input_fn` raises an end-of-input
                exception (`OutOfRangeError` or `StopIteration`).
            dtypes:  A nested structure of tf.DType objects corresponding to
                each component of an element yielded by generator.

        """
        self.dtypes = dtypes

        self.read_fn = read_fn

    def get_inputs(self,
                   file_references,
                   mode,
                   example_shapes=None,
                   shuffle_cache_size=100,
                   batch_size=4,
                   params=None):
        """
        Function to provide the input_fn for a tf.Estimator.

        Args:
            file_references: An array like structure that holds the reference
                to the file to read. It can also be None if not needed.
            mode: A tf.estimator.ModeKeys. It is passed on to `read_fn` to
                trigger specific functions there.
            example_shapes (optional): A nested structure of lists or tuples
                corresponding to the shape of each component of an element
                yielded by generator.
            shuffle_cache_size (int, optional): An `int` determining the
                number of examples that are held in the shuffle queue.
            batch_size (int, optional): An `int` specifying the number of
                examples returned in a batch.
            params (dict, optional): A `dict` passed on to the `read_fn`.

        Returns:
            function: a handle to the `input_fn` to be passed the relevant
                tf estimator functions.
            tf.train.SessionRunHook: A hook to initialize the queue within
                the dataset.
        """
        iterator_initializer_hook = IteratorInitializerHook()

        def train_inputs():
            def f():
                def clean_ex(ex, compare):
                    # Clean example dictionary by recursively deleting
                    # non-relevant entries. However, this does not look into
                    # dictionaries nested into lists
                    for k in list(ex.keys()):
                        if k not in list(compare.keys()):
                            del ex[k]
                        elif isinstance(ex[k], dict) and isinstance(compare[k], dict):
                            clean_ex(ex[k], compare[k])
                        elif (isinstance(ex[k], dict) and not isinstance(compare[k], dict)) or \
                             (not isinstance(ex[k], dict) and isinstance(compare[k], dict)):
                            raise ValueError('Entries between example and '
                                             'dtypes incompatible for key {}'
                                             ''.format(k))
                        elif (isinstance(ex[k], list) and not isinstance(compare[k], list)) or \
                            (not isinstance(ex[k], list) and isinstance(compare[k], list)) or \
                                (isinstance(ex[k], list) and isinstance(compare[k], list) and not
                                    len(ex[k]) == len(compare[k])):
                            raise ValueError('Entries between example and '
                                             'dtypes incompatible for key {}'
                                             ''.format(k))
                    for k in list(compare):
                        if k not in list(ex.keys()):
                            raise ValueError('Key {} not found in ex but is '
                                             'present in dtypes. Found keys: '
                                             '{}'.format(k, ex.keys()))
                    return ex

                fn = self.read_fn(file_references, mode, params)
                # iterate over all entries - this loop is terminated by the
                # tf.errors.OutOfRangeError or StopIteration thrown by the
                # read_fn
                while True:
                    try:
                        ex = next(fn)

                        if ex.get('labels') is None:
                            ex['labels'] = None

                        if not isinstance(ex, dict):
                            raise ValueError('The read_fn has to return '
                                             'dictionaries')

                        ex = clean_ex(ex, self.dtypes)
                        yield ex
                    except (tf.errors.OutOfRangeError, StopIteration):
                        raise
                    except Exception as e:
                        print('got error `{} from `_read_sample`:'.format(e))
                        print(traceback.format_exc())
                        raise

            dataset = tf.data.Dataset.from_generator(
                f, self.dtypes, example_shapes)
            dataset = dataset.repeat(None)
            dataset = dataset.shuffle(shuffle_cache_size)
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(1)

            iterator = dataset.make_initializable_iterator()
            next_dict = iterator.get_next()

            # Set runhook to initialize iterator
            iterator_initializer_hook.iterator_initializer_func = \
                lambda sess: sess.run(iterator.initializer)

            # Return batched (features, labels)
            return next_dict['features'], next_dict.get('labels')

        # Return function and hook
        return train_inputs, iterator_initializer_hook

    def serving_input_receiver_fn(self, placeholder_shapes):
        """Build the serving inputs.

        Args:
            placeholder_shapes: A nested structure of lists or tuples
                corresponding to the shape of each component of the feature
                elements yieled by the read_fn.

        Returns:
            function: A function to be passed to the tf.estimator.Estimator
            instance when exporting a saved model with estimator.export_savedmodel.
        """

        def f():
            inputs = {k: tf.placeholder(
                shape=[None] + list(placeholder_shapes['features'][k]),
                dtype=self.dtypes['features'][k]) for k in list(self.dtypes['features'].keys())}

            return tf.estimator.export.ServingInputReceiver(inputs, inputs)
        return f
