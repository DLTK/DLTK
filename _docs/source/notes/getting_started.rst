Getting Started
===============

.. toctree::
   :hidden:

   module
   model
   reader
   usage

Explain training of first model -> look at mnist example
DLTK uses a plug&play approach to deep learning. :doc:`module` implement layers of Neural Networks. You can combine
them to build neural networks as :doc:`model`. Using :doc:`reader` you can read the data for training your network.
Everything uses standard `Tensorflow <https://tensorflow.org>`_ code and is easily extensible. Apart from those building
blocks you still need to code your basic training loop to have your network learn something.

An easy example of using DLTK is given in the MNIST classification example. This example uses the LeNet5 architecture
for digit classification. You can read the data with the Tensorflow handlers
::
  from tensorflow.examples.tutorials.mnist import input_data
  mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

and build the network from DLTK
::
  from dltk.models.classification.lenet import LeNet5
  net = LeNet5(num_classes=10)

This does not provide inputs to the network yet. To pass inputs through and receive the outputs of the network you have
to call it first. For this we build tf.Placeholders for data input and pass them through the network:
::
  import tensorflow as tf
  # Create placeholders to feed input data during execution
  xp = tf.placeholder(tf.float32, shape=[None, 784])
  yp = tf.placeholder(tf.int32, shape=[None, ])

  # Reshape the input images x_in from [None, 784] to [None, 28, 28, 1], where the tensor dimensions are [batch_size,x,y,channels]
  x_in = tf.reshape(xp, [-1, 28, 28, 1])

  logits_ = net(x_in)['logits']

For training we need to provide a measure of how wrong our model is. Here we use the crossentropy error defined by
Tensorflow
::
  crossentropy_ = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_, labels=yp)
  loss_ = tf.reduce_mean(crossentropy_, name='crossentropy')

and minimise this loss with the ADAM optimizer
::
  train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss_)

We can also build validation metrics like the accuracy using a network in test mode
::
  # Create additional ops to visualise the network output and track the training steps
  y_hat_ = net(x_in, is_training=False)['y_']
  val_acc_ = tf.reduce_mean(tf.cast(tf.equal(tf.cast(yp, tf.int32), tf.cast(y_hat_, tf.int32)), tf.float32))

We finally build placeholders for tracking the accuracy and the loss and build a Tensorflow Session:
::
  loss_moving = []
  acc_moving = []

  s = tf.Session()

Using the operations defined we can build a training loop and train the network
::
  for step in range(1000):
      # Get a batch of training input pairs of x (image) and y (label)
      batch = mnist.train.next_batch(100)

      # Run the training op and the loss
      _, logits, loss = s.run([train_op, logits_, loss_], feed_dict={xp: batch[0], yp: batch[1]})
      loss_moving.append(loss)

      # Compute the test accuracy
      val_acc = s.run(val_acc_, feed_dict={xp: mnist.test.images, yp: mnist.test.labels})
      acc_moving.append(val_acc)

And visualise the training by plotting the loss and accuracy curves
::
  import matplotlib.pyplot as plt
  f, axarr = plt.subplots(1, 3, figsize=(16,4))

  axarr[0].imshow(np.reshape(batch[0], [-1, 28, 28])[-1], cmap='gray', vmin=0, vmax=1)
  axarr[0].set_title('Input x; Prediction = {}; Truth = {};'.format(np.argmax(logits[-1,]), batch[1][-1,]))
  axarr[0].axis('off')

  axarr[1].semilogy(loss_moving)
  axarr[1].set_title('Training loss')
  axarr[1].axis('on')

  axarr[2].semilogy(acc_moving)
  axarr[2].set_title('Test acc')
  axarr[2].axis('on')

  plt.show()