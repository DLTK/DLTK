from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
import SimpleITK as sitk


class AbstractReader(object):
    """Abstract reader

    Abstract reader class for data I/O. Provides the queue handling and wraps the specific reader functions to adapt
    given data types.

    """
    def __init__(self, dtypes, dshapes, name='reader'):
        """AbstractReader

        Construcsts an abstract reader

        Parameters
        ----------
        dtypes : list or tuple
            list of dtypes for the tensors in the queue
        dshapes : list or tuple
            list of shapes for the tensors in the queue
        name : string
            name of the reader, used for the name scope
        """
        self.name = name
        self.dtypes = dtypes
        self.dshapes = dshapes

        self.__call__.__func__.__doc__ = self._create_queue.__doc__

    def _preprocess(self, data):
        """ placeholder for the preprocessing of reader subclasses """
        return data

    def _augment(self, data):
        """ placeholder for the augmentation of reader subclasses """
        return data

    def _read_sample(self, id_queue, **kwargs):
        """ placeholder for the reading of independent samples of reader subclasses """
        raise NotImplementedError('Abstract reader - not implemented')

    @staticmethod
    def _map_dtype(dtype):
        """ helper function to map tf data types to np data types """
        if dtype == tf.float32:
            return np.float32
        elif dtype == tf.int32:
            return np.int32
        elif dtype == tf.float64:
            return np.float64
        elif dtype == tf.int64:
            return np.int64
        else:
            raise Exception('Dtype not handled')

    def _create_queue(self, id_list, shuffle=True, batch_size=16, num_readers=1, min_queue_examples=64,
                      capacity=128, **kwargs):
        """ Builds the data queue using the '_read_sample' function

        Parameters
        ----------
        id_list : list or tuple
            list of examples to read. This can be a list of files or a list of ids or something else the read function
            understands
        shuffle : bool
            flag to toggle shuffling of examples
        batch_size : int
        num_readers : int
            number of readers to spawn to fill the queue. this is used for multi-threading and should be tuned
            according to the specific problem at hand and hardware available
        min_queue_examples : int
            minimum number of examples currently in the queue. This can be tuned for more preloading of the data
        capacity : int
            maximum number of examples the queue will hold. a lower number needs less memory whereas a higher number
            enables better mixing of the examples
        kwargs :
            additional arguments to be passed to the reader function

        Returns
        -------
        list
            list of tensors representing a batch from the queue

        """
        with tf.name_scope(self.name):
            # Create filename_queue
            id_tensor = tf.convert_to_tensor(id_list, dtype=tf.string)

            id_queue = tf.train.slice_input_producer([id_tensor],  capacity=16, shuffle=shuffle)

            if num_readers < 1:
                raise ValueError('Please make num_readers at least 1')

            # Build a FIFO or a shuffled queue
            if shuffle:
                examples_queue = tf.RandomShuffleQueue(
                    capacity=capacity,
                    min_after_dequeue=min_queue_examples,
                    dtypes=self.dtypes)
            else:
                examples_queue = tf.FIFOQueue(
                    capacity=capacity,
                    dtypes=self.dtypes)

            if num_readers > 1:
                # Create multiple readers to populate the queue of examples.
                enqueue_ops = []
                for _ in range(num_readers):
                    ex = self._read_wrapper(id_queue, **kwargs)
                    enqueue_ops.append(examples_queue.enqueue_many(ex))

                tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))

                ex = examples_queue.dequeue()
            else:
                # Use a single reader for population
                ex = self._read_wrapper(id_queue, **kwargs)

            # create a batch_size tensor with default shape, to keep the downstream graph flexible
            batch_size_tensor = tf.placeholder_with_default(batch_size, shape=[], name='batch_size_ph')

            # batch the read examples
            ex_batch = tf.train.batch(
                ex,
                batch_size=batch_size_tensor,
                enqueue_many=True,
                capacity=2 * num_readers * batch_size)

            return ex_batch

    def _read_wrapper(self, id_queue, **kwargs):
        """ Wrapper for the '_read_sample' function

        Wraps the 'read_sample function and handles tensor shapes and data types

        Parameters
        ----------
        id_queue : list
            list of tf.Tensors from the id_list queue. Provides an identifier for the examples to read.
        kwargs :
            additional arguments for the '_read_sample function'

        Returns
        -------
        list
            list of tf.Tensors read for this example

        """
        def f(id_queue):
            """ Wrapper for the python function

            Handles the data types of the py_func

            Parameters
            ----------
            id_queue : list
            list of tf.Tensors from the id_list queue. Provides an identifier for the examples to read.

            Returns
            -------
            list
                list of things just read
            """
            ex = self._read_sample(id_queue, **kwargs)

            # eventually fix data types of read objects
            tensors = []
            for t, d in zip(ex, self.dtypes):
                if isinstance(t, np.ndarray):
                    tensors.append(t.astype(self._map_dtype(d)))
                elif isinstance(t, (float, int)):
                    if d is tf.float32 and isinstance(t, int):
                        print('Warning: Losing accuracy by converting int to float')
                    tensors.append(self._map_dtype(d)(t))
                elif isinstance(t, bool):
                    tensors.append(t)
                else:
                    raise Exception('Not sure how to interpret "{}"'.format(type(t)))
            return tensors

        ex = tf.py_func(f, [id_queue], self.dtypes)
        tensors = []
        # set shape of tensors for downstream inference of shapes
        for t, s in zip(ex, self.dshapes):
            t.set_shape([None] + list(s))
            tensors.append(t)
        return tensors

    def __call__(self, *args, **kwargs):
        return self._create_queue(*args, **kwargs)


class SimpleSITKReader(AbstractReader):
    """SimpleSITKReader

    Simple reader class to read sitk files by file path

    """
    def __init__(self, dtypes, dshapes, name='simplesitkreader'):
        super(SimpleSITKReader, self).__init__(dtypes, dshapes, name=name)

    def _read_sample(self, id_queue, **kwargs):
        path_list = id_queue[0]

        data = []

        for p, d in zip(list(path_list), self.dtypes):
            if isinstance(p, str):
                # load image etc
                sample = sitk.GetArrayFromImage(sitk.ReadImage(p))
                data.append(sample.astype(self._map_dtype(d)))
            elif isinstance(p, (float, int)):
                # load label
                if d is tf.float32 and isinstance(p, int):
                    print('Warning: Losing accuracy by converting int to float')
                data.append(self._map_dtype(d)(p))
            else:
                raise Exception('Not sure how to interpret "{}"'.format(p))

        data = self._preprocess(data)
        data = self._augment(data)

        return data