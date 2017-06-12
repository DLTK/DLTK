from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
from collections import namedtuple

import cifar_input as reader
import numpy as np
import tensorflow as tf

import dltk.core.modules as modules
from dltk.models.classification.resnet import ResNet

# Training parameters
training_params = namedtuple('training_params',
                             'max_steps, batch_size, save_summary_sec, save_model_sec, steps_eval')
training_params.__new__.__defaults__ = (2e6, 128, 10, 600, 100)
tps = training_params()



def train(args):
    """
        Complete training and validation script. Additionally, saves inference model, trained weights and summaries.

        Parameters
        ----------
        args : argparse.parser object
            contains all necessary command line arguments

        Returns
        -------
    """

    if not args.resume:
        os.system("rm -rf %s" % args.save_path)
        os.system("mkdir -p %s" % args.save_path)
    else:
        print('Resuming training')

    num_classes = 10
    batch_size = 128

    # Define and build the network graph
    net = ResNet(num_classes, strides=[[1, 1], [1, 1], [2, 2], [2, 2]])

    # Parse the csv files and define input ops for training and validation I/O
    print('Loading data from {}'.format(args.data_dir))
    x_train, y_train = reader.build_input('cifar10',
                                          os.path.join(args.data_dir, 'data_batch*'),
                                          batch_size,
                                          'train')

    # Define training metrics and optimisation ops
    train_net = net(x_train)
    train_logits_ = train_net['logits']
    train_pred_ = train_net['y_']
    train_truth_ = y_train

    train_acc_ = tf.reduce_mean(tf.cast(tf.equal(tf.cast(train_truth_, tf.int32),
                                                 tf.cast(train_pred_, tf.int32)), tf.float32))
    modules.scalar_summary(train_acc_, 'train/acc', collections=['losses', 'metrics'])

    ce = modules.sparse_crossentropy(train_logits_, train_truth_, name='train/loss',
                                     collections=['losses', 'training'])
    l2 = modules.l2_regularization(net.get_variables(tf.GraphKeys.WEIGHTS), 0.0002, name='train/l2',
                                   collections=['training', 'regularization'])
    train_loss_ = ce + l2

    lr_placeholder = tf.placeholder(tf.float32)
    train_op_ = tf.train.MomentumOptimizer(lr_placeholder, 0.9).minimize(train_loss_)

    train_summaries = tf.summary.merge([tf.summary.merge_all('training'), ] +
                                       [tf.summary.histogram(var.name, var) for var
                                        in net.get_variables(tf.GraphKeys.MOVING_AVERAGE_VARIABLES)])

    if args.run_validation:
        X_test, Y_test = reader.build_input('cifar10',
                                            os.path.join(args.data_dir, 'test_batch*'),
                                            100,
                                            'eval')
        # Define validation outputs
        val_net = net(X_test, is_training=False)
        val_logits_ = val_net['logits']
        val_pred_ = val_net['y_']
        val_truth_ = Y_test

        val_loss_ = modules.sparse_crossentropy(val_logits_, val_truth_, collections=['losses', 'validation'])
        val_acc_ = tf.reduce_mean(tf.cast(tf.equal(tf.cast(val_truth_, tf.int32), tf.cast(val_pred_, tf.int32)), tf.float32))

    # Define and setup a training supervisor
    global_step = tf.Variable(0, name='global_step', trainable=False)
    sv = tf.train.Supervisor(logdir=args.save_path,
                             is_chief=True,
                             summary_op=None,
                             save_summaries_secs=tps.save_summary_sec,
                             save_model_secs=tps.save_model_sec,
                             global_step=global_step)

    s = sv.prepare_or_wait_for_session(config=tf.ConfigProto())

    # Main training loop
    step = s.run(global_step) if args.resume else 0
    while not sv.should_stop():

        if step < 40000:
            lr = 0.1
        elif step < 60000:
            lr = 0.01
        elif step < 80000:
            lr = 0.001
        else:
            lr = 0.0001
            
        # Run the training op
        _ = s.run(train_op_, feed_dict={lr_placeholder: lr})

        # Evaluation of training and validation data
        if step % tps.steps_eval == 0:
            (train_loss, train_acc, train_pred, train_truth, t_sum) = s.run(
                [train_loss_, train_acc_, train_pred_, train_truth_, train_summaries])
            sv.summary_computed(s, t_sum, global_step=step)

            print("\nEval step= {:d}".format(step))
            print("Train: Loss= {:.6f}; Acc {:.6f}".format(train_loss, train_acc))

            # Evaluate all validation data
            if args.run_validation:
                all_loss = [];
                all_acc = [];
                for _ in range(50):
                    (val_loss, val_pred, val_truth, val_acc) = s.run([val_loss_, val_pred_, val_truth_, val_acc_])

                    all_loss.append(val_loss)
                    all_acc.append(val_acc)

                mean_loss = np.mean(all_loss, axis=0)
                mean_acc = np.mean(all_acc, axis=0)

                sv.summary_computed(s, modules.scalar_summary(mean_loss, 'val/loss'), global_step=step)
                sv.summary_computed(s, modules.scalar_summary(mean_acc, 'val/acc'), global_step=step)

                print("Valid: Loss= {:.6f}; Acc {:.6f}".format(mean_loss, mean_acc))

        # Stopping condition
        if step >= tps.max_steps and tps.max_steps > 0:
            print('Run %d steps of %d steps - stopping now' % (step, tps.max_steps))
            break

        step += 1


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CIFAR10 classification training script')
    parser.add_argument('--run_validation', default=True)

    parser.add_argument('--gpu_memory', default=None)
    parser.add_argument('--resume', default=True, action='store_true')
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--cuda_devices', '-c', default='0')

    parser.add_argument('--data_dir', default='../../data/cifar-10-batches-bin/')

    parser.add_argument('--save_path', '-p',
                        default='/tmp/cifar10')

    args = parser.parse_args()
    if args.verbose:
        tf.logging.set_verbosity(tf.logging.INFO)
    else:
        tf.logging.set_verbosity(tf.logging.ERROR)

    # GPU allocation options
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    train(args)
