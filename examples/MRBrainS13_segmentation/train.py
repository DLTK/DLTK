from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
from collections import namedtuple

import mrbrains_reader as reader
import numpy as np
import pandas as pd
import tensorflow as tf

import dltk.core.modules as modules
from dltk.models.segmentation.unet import ResUNET
from dltk.core import batcher as batcher
from dltk.core import metrics as metrics

# Training parameters
training_params = namedtuple('training_params',
                             'max_steps, batch_size, save_summary_sec, save_model_sec, steps_eval')
training_params.__new__.__defaults__ = (1e5, 16, 10, 600, 100)
tps = training_params()


def eval_mrbrains_metrics(y_, y):
    """
        Re-map the labels to CSF, GM, WM and evaluate the metrics for MRBrainS13:
        
        from: 
        1. Cortical gray matter
        2. Basal ganglia
        3. White matter
        4. White matter lesions
        5. Cerebrospinal fluid in the extracerebral space
        6. Ventricles
        7. Cerebellum
        8. Brainstem

        to:
        
        1. CSF
        2. GM
        3. WM

        Parameters
        ----------
        y : np array
            a valdiation label map
        
        y_ : np array
            a prediction label map
        
        Returns
        -------
        mrbrains_summary : a tensorboard summary
    """

    y_mapped_ = np.zeros_like(y_)
    y_mapped_[y_==1] = 2
    y_mapped_[y_==2] = 2
    y_mapped_[y_==3] = 3
    y_mapped_[y_==4] = 3
    y_mapped_[y_==5] = 1
    y_mapped_[y_==6] = 1
    
    y_mapped = np.zeros_like(y)
    y_mapped[y==1] = 2
    y_mapped[y==2] = 2
    y_mapped[y==3] = 3
    y_mapped[y==4] = 3
    y_mapped[y==5] = 1
    y_mapped[y==6] = 1
    
    # remove voxel locations that are Cerebellum or Brainstem from evaluation
    y_mapped_[y==7] = 0
    y_mapped_[y==8] = 0
    
    # compute metrics between the mapped labelmaps:
    dscs = metrics.dice(y_mapped_, y_mapped, num_classes=4)
    avds = metrics.abs_vol_difference(y_mapped_, y_mapped, num_classes=4)
    
    return dscs, avds
    

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

    num_classes = 9
    num_channels = 3
    batch_size = 4

    g = tf.Graph()
    with g.as_default():
        
        # Set a seed 
        np.random.seed(1337)
        tf.set_random_seed(1337)

        # Build the network graph
        net = ResUNET(num_classes, num_residual_units=3, filters=[32, 64, 128, 256],
                      strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2]])

        # Parse the csv files 
        print('Loading training file names from %s' % args.train_csv)
        train_files = pd.read_csv(args.train_csv, dtype=object).as_matrix()

        # I/O ops for training and validation via a custom mrbrains_reader
        x_train, y_train = reader.MRBrainsReader([tf.float32, tf.int32], [[24, 64, 64, 3], [24, 64, 64]], name='train_queue')(
            train_files, batch_size=batch_size, n_examples=18, min_queue_examples=batch_size * 2, capacity=batch_size * 4)

        # Training metrics and optimisation ops
        train_net = net(x_train)
        train_logits_ = train_net['logits']
        train_pred_ = train_net['y_']
        train_truth_ = y_train

        # Add image summaries and a summary op
        modules.image_summary(x_train, 'train_img', ['training'])
        modules.image_summary(tf.expand_dims(tf.to_float(train_pred_) / num_classes, axis=-1), 'train_pred', ['training'])
        modules.image_summary(tf.expand_dims(tf.to_float(y_train) / num_classes, axis=-1), 'train_lbl', ['training'])

        train_summaries = tf.summary.merge([tf.summary.merge_all('training'), ] +
                                           [tf.summary.histogram(var.name, var) for var
                                            in net.get_variables(tf.GraphKeys.MOVING_AVERAGE_VARIABLES)])

        
        # Add crossentropy loss and regularisation
        ce = modules.sparse_crossentropy(train_logits_, train_truth_, name='train/loss',
                                         collections=['losses', 'training'])
        l1 = modules.l1_regularization(net.get_variables(tf.GraphKeys.BIASES), 1e-5, name='train/l1',
                                         collections=['training', 'regularization'])
        l2 = modules.l2_regularization(net.get_variables(tf.GraphKeys.WEIGHTS), 1e-4, name='train/l2',
                                       collections=['training', 'regularization'])
        
        train_loss_ = ce + l2 
        
        # Create a learning rate placeholder for scheduling and choose an optimisation
        lr_placeholder = tf.placeholder(tf.float32)
        train_op_ = tf.train.MomentumOptimizer(lr_placeholder, 0.9).minimize(train_loss_)

        # Set up ops for validation 
        if args.run_validation:
            print('Loading validation file names from %s' % args.val_csv)
            val_files = pd.read_csv(args.val_csv, dtype=str).as_matrix()
            val_reader = reader.MRBrainsReader([tf.float32, tf.int32], [[48, 240, 240, 3], [48, 240, 240]], name='val_queue')
            val_read_func = lambda x: val_reader._read_sample(x, is_training=False)

            # Reuse the training model for validation inference and replace inputs with placeholders
            x_placeholder = tf.placeholder(tf.float32, shape=[None, None, None, None, num_channels], name='x_placeholder')
            y_placeholder = tf.placeholder(tf.int32, shape=[None, None, None, None], name='y_placeholder')

            val_net = net(x_placeholder, is_training=False)
            val_logits_ = val_net['logits']
            val_pred_ = val_net['y_']
            val_truth_ = y_placeholder

            val_loss_ = modules.sparse_crossentropy(val_logits_, val_truth_, collections=['losses', 'validation'])

        # Define and set up a training supervisor, handling queues and logging for tensorboard
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
            
            # Each step run the training op with a learning rate schedule
            lr = 0.001 if step < 40000 else 0.0001
            _ = s.run(train_op_, feed_dict={lr_placeholder: lr})

            # Evaluation of training and validation data
            if step % tps.steps_eval == 0:
                (train_loss, train_pred, train_truth, t_sum) = s.run([train_loss_, train_pred_, train_truth_, train_summaries])
                dscs, avds = eval_mrbrains_metrics(train_pred, train_truth)
                
                # Build custom metric summaries and add them to tensorboard
                sv.summary_computed(s, t_sum, global_step=step)
                sv.summary_computed(s, modules.scalar_summary({n: val for n, val in zip(['CSF', 'GM', 'WM'], dscs[1:])},
                                                              'train/dsc'), global_step=step)
                sv.summary_computed(s, modules.scalar_summary({n: val for n, val in zip(['CSF', 'GM', 'WM'], avds[1:])},
                                                              'train/avd'), global_step=step)
                
                print("\nEval step= {:d}".format(step))
                print("Train: Loss= {:.6f}; DSC= {:.4f} {:.4f} {:.4f}, AVD= {:.4f} {:.4f} {:.4f} ".format(train_loss,  dscs[1], dscs[2], dscs[3], avds[1], avds[2], avds[3]))

                # Run inference on all validation data (here, just one dataset) and compute mean performance metrics
                if args.run_validation:
                    all_loss = []; all_dscs = []; all_avds = [];
                    for f in val_files:
                        val_x, val_y = val_read_func([f])
                        val_x = val_x[np.newaxis, :]
                        val_y = val_y[np.newaxis, :]
                        (val_loss, val_pred, val_truth) = s.run([val_loss_, val_pred_, val_truth_], 
                                                                feed_dict={x_placeholder: val_x, y_placeholder: val_y})
                        
                        dscs, avds = eval_mrbrains_metrics(val_pred, val_truth)

                        all_loss.append(val_loss)
                        all_dscs.append(dscs)
                        all_avds.append(avds)

                    mean_loss = np.mean(all_loss, axis=0)
                    mean_dscs = np.mean(all_dscs, axis=0)
                    mean_avds = np.mean(all_avds, axis=0)

                    # Add them to tensorboard as image and metrics summaries
                    sv.summary_computed(s, modules.image_summary(val_x[0], 'val_img'), global_step=step)
                    sv.summary_computed(s, modules.image_summary(val_y[0, :, :, :, np.newaxis]  / num_classes, 'val_lbl'), global_step=step)
                    sv.summary_computed(s, modules.image_summary(val_pred[0, :, :, :, np.newaxis] / num_classes,
                                                                 'val_pred'), global_step=step)

                    sv.summary_computed(s, modules.scalar_summary(mean_loss, 'val/loss'), global_step=step)
                    sv.summary_computed(s, modules.scalar_summary({n: val for n, val in zip(['CSF', 'GM', 'WM'],
                                                                                            mean_dscs[1:])},
                                                                  'val/dsc'), global_step=step)
                    sv.summary_computed(s, modules.scalar_summary({n: val for n, val in zip(['CSF', 'GM', 'WM'],
                                                                                            mean_avds[1:])},
                                                                  'val/avd'), global_step=step)

                    print("Valid: Loss= {:.6f}; DSC= {:.4f} {:.4f} {:.4f}, AVD= {:.4f} {:.4f} {:.4f} ".format(mean_loss, mean_dscs[1], mean_dscs[2], mean_dscs[3], mean_avds[1], mean_avds[2], mean_avds[3]))   

            # Stopping condition
            if step >= tps.max_steps and tps.max_steps > 0:
                print('Run %d steps of %d steps - stopping now' % (step, tps.max_steps))
                break

            step += 1

            
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MRBrainS13 segmentation training script')
    parser.add_argument('--run_validation', default=True)

    parser.add_argument('--gpu_memory', default=None)
    parser.add_argument('--resume', default=True, action='store_true')
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--cuda_devices', '-c', default='0')

    parser.add_argument('--train_csv', default='train.csv')
    parser.add_argument('--val_csv', default='val.csv')

    parser.add_argument('--save_path', '-p', default='/tmp/MRBrainS13_segmentation')

    args = parser.parse_args()
    if args.verbose:
        tf.logging.set_verbosity(tf.logging.INFO)
    else:
        tf.logging.set_verbosity(tf.logging.ERROR)

    # GPU allocation options
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    train(args)
