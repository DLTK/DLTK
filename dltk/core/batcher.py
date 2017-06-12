from __future__ import print_function

import tensorflow as tf

def get_inputs(filenames, reader, mode='train', batch_size=16, num_readers=1, name_scope='batch_processing',
               input_queue_memory_factor=4, examples_per_shard=32):
    
    with tf.name_scope(name_scope):
        
        # Create filename_queue
        filename_tensor = tf.convert_to_tensor(filenames, dtype=tf.string)
        
        if mode=='train':
            filename_queue = tf.train.slice_input_producer([filename_tensor],
                                                           shuffle=True,
                                                           capacity=16)
            read_func = reader.read
            ex_size = reader.get_example_shapes()
        else:
            filename_queue = tf.train.slice_input_producer([filename_tensor],
                                                           shuffle=False,
                                                           capacity=1)
            read_func = lambda  x: reader.read(x, is_test=True)
            ex_size = reader.get_example_shapes(is_test=True)
        if num_readers < 1:
            raise ValueError('Please make num_readers at least 1')
    
        # Size the random shuffle queue to balance between good global
        # mixing (more examples) and memory use (fewer examples):
        # examples_per_shard: examples per read cycle returned from the reader
        # input_queue_memory_factor: multiplier of retained examples in the batch queue 
        examples_per_shard = ex_size[0][0]
        min_queue_examples = examples_per_shard * input_queue_memory_factor
        
        # Setting up reader queues for training and testing
        if mode=='train':
            examples_queue = tf.RandomShuffleQueue(
                capacity=min_queue_examples + 3 * batch_size,
                min_after_dequeue=min_queue_examples,
                dtypes=[tf.float32, tf.int32])
        else:
            examples_queue = tf.FIFOQueue(
                capacity=examples_per_shard + 3 * batch_size,
                dtypes=[tf.float32, tf.int32])
            
        # Create multiple readers to populate the queue of examples.
        if num_readers > 1:
            enqueue_ops = []
            for _ in range(num_readers):
                ei, el = read_func(filename_queue)
                enqueue_ops.append(examples_queue.enqueue_many([ei, el]))

            tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))

            ex_img, ex_lbl = examples_queue.dequeue()

            ex_img = tf.reshape(ex_img, ex_size[0])
            ex_lbl = tf.reshape(ex_lbl, ex_size[1])
            ex_img.set_shape(ex_size[0])
            ex_lbl.set_shape(ex_size[1])
        else:
            ex_img, ex_lbl = read_func(filename_queue)
        # create a batch_size tensor with default shape, to keep the downstream graph flexible
        batch_size_tensor = tf.placeholder_with_default(batch_size, shape=[], name='batch_size_ph')

        #ex_img = tf.Print(ex_img, [ex_img.shape], message='shape')

        img_batch, lbl_batch = tf.train.batch_join(
        [[ex_img, ex_lbl]],
        batch_size=batch_size_tensor,
        enqueue_many=True,
        capacity=2 * num_readers * batch_size)

        return img_batch, lbl_batch