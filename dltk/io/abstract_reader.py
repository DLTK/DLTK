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
    """Wrapper for dataset generation given a read function and a save function"""
    def __init__(self, read_fn, save_fn, dtypes):
        """

        Args:
            read_fn: Input function returning features which is a dictionary of
        string feature name to `Tensor` or `SparseTensor`. If it returns a
        tuple, first item is extracted as features. Prediction continues until
        `input_fn` raises an end-of-input exception (`OutOfRangeError` or
        `StopIteration`).
            save_fn:
            dtypes:
        """
        self.dtypes = dtypes

        self.read_fn = read_fn
        self.save = save_fn

    def get_inputs(self, file_references, mode, example_shapes=None, shuffle_cache_size=100, batch_size=4,
                   params=None):
        """
        Function to provide the input_fn for a tf.Estimator.

        Args:
            file_references: An array like structure that holds the reference to the file to read
            mode: A tf.estimator.ModeKeys. It is passed on to `read_fn` to trigger specific functions there.
            example_shapes:
            shuffle_cache_size: An `int` determining the number of examples that are held in the shuffle queue.
            batch_size: An `int` specifying the number of examples returned in a batch.
            params: A `dict` passed on to the `read_fn`.

        Returns:
            tf.Tensor, tf.train.SessionRunHook
        """
        iterator_initializer_hook = IteratorInitializerHook()

        def train_inputs():
            def f():
                fn = self.read_fn(file_references, mode, params)
                while True:
                    try:
                        ex = next(fn)
                        yield ex
                    except (tf.errors.OutOfRangeError, StopIteration):
                        raise
                    except Exception as e:
                        print('got error `{} from `_read_sample`:'.format(e))
                        print(traceback.format_exc())
                        raise

            dataset = tf.data.Dataset.from_generator(f,
                                                             self.dtypes, example_shapes)
            dataset = dataset.repeat(None)
            dataset = dataset.shuffle(shuffle_cache_size)
            dataset = dataset.batch(batch_size)

            iterator = dataset.make_initializable_iterator()
            next_dict = iterator.get_next()

            # Set runhook to initialize iterator
            iterator_initializer_hook.iterator_initializer_func = lambda sess: sess.run(iterator.initializer)

            # Return batched (features, labels)
            return next_dict['features'], next_dict.get('labels')

        # Return function and hook
        return train_inputs, iterator_initializer_hook

    def serving_input_receiver_fn(self, placeholder_shapes):
        """Build the serving inputs."""
        # The outer dimension (None) allows us to batch up inputs for
        # efficiency. However, it also means that if we want a prediction
        # for a single instance, we'll need to wrap it in an outer list.
        def f():
            inputs = {k: tf.placeholder(shape=[None,] + list(placeholder_shapes['features'][k]),
                                        dtype=self.dtypes['features'][k])
                      for k in self.dtypes['features'].keys()}

            #pl = tf.placeholder(shape=[None] + EXAMPLE_SIZE + [3], dtype=tf.float32)
            #inputs = {"x": pl}
            return tf.estimator.export.ServingInputReceiver(inputs, inputs)
        return f