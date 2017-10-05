from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import json
import numpy as np
import tensorflow as tf
import model_dcae
import utils

FLAGS = tf.app.flags.FLAGS

# path parameters
tf.app.flags.DEFINE_string("train_data", "train_data_train_100_label_100.npy",
                           "training data name")
tf.app.flags.DEFINE_string("valid_data", "test_data_train_100_label_100.npy",
                           "validation data name")
tf.app.flags.DEFINE_string('data_dir', '/home/xdex/Desktop/traffic_flow_detection/taipei/training_data/old_Taipei_data/vd_base/',
                           "data directory")
# tf.app.flags.DEFINE_string('data_dir', '/home/xdex/Desktop/traffic_flow_detection/taipei/training_data/new_raw_data/vd_base/',
#                            "data directory")
tf.app.flags.DEFINE_string('checkpoints_dir', 'v1/checkpoints/',
                           "training checkpoints directory")
tf.app.flags.DEFINE_string('log_dir', 'v1/log/',
                           "summary directory")
tf.app.flags.DEFINE_string('restore_path', None,
                           "path of saving model eg: checkpoints/model.ckpt-5")
# data augmentation and corruption
tf.app.flags.DEFINE_integer('aug_ratio', 4,
                            "the ratio of data augmentation")
tf.app.flags.DEFINE_integer('corrupt_ratio', 0.10,
                            "the amount of corrupted data")
# training parameters
FILTER_NUMBERS = [32, 64, 128]
FILTER_STRIDES = [1,  2,   2]
tf.app.flags.DEFINE_integer('batch_size', 512,
                            "mini-batch size")
tf.app.flags.DEFINE_integer('total_epoches', 500,
                            "total training epoches")
tf.app.flags.DEFINE_integer('save_freq', 25,
                            "number of epoches to saving model")
tf.app.flags.DEFINE_float('learning_rate', 1e-4,
                          "learning rate of AdamOptimizer")
# tf.app.flags.DEFINE_integer('num_gpus', 2,
#                             "multi gpu")
# flags
tf.app.flags.DEFINE_bool('if_label_normed', True,
                         "if label data normalized, we need to recover its scale back before compute loss")
tf.app.flags.DEFINE_bool('if_mask_only', True,
                         "if the loss computed on mask only")


class TrainingConfig(object):
    """
    Training config
    """

    def __init__(self, filter_numbers, filter_strides, input_shape):
        self.filter_numbers = filter_numbers
        self.filter_strides = filter_strides
        self.input_shape = input_shape
        self.data_dir = FLAGS.data_dir
        self.checkpoints_dir = FLAGS.checkpoints_dir
        self.log_dir = FLAGS.log_dir
        self.restore_path = FLAGS.restore_path
        self.aug_ratio = FLAGS.aug_ratio
        self.corrupt_ratio = FLAGS.corrupt_ratio
        self.batch_size = FLAGS.batch_size
        self.total_epoches = FLAGS.total_epoches
        self.save_freq = FLAGS.save_freq
        self.learning_rate = FLAGS.learning_rate
        self.if_label_normed = FLAGS.if_label_normed
        self.if_mask_only = FLAGS.if_mask_only

    def show(self):
        print("filter_numbers:", self.filter_numbers)
        print("filter_strides:", self.filter_strides)
        print("input_shape:", self.input_shape)
        print("data_dir:", self.data_dir)
        print("checkpoints_dir:", self.checkpoints_dir)
        print("log_dir:", self.log_dir)
        print("restore_path:", self.restore_path)
        print("aug_ratio:", self.aug_ratio)
        print("corrupt_ratio:", self.corrupt_ratio)
        print("batch_size:", self.batch_size)
        print("total_epoches:", self.total_epoches)
        print("save_freq:", self.save_freq)
        print("learning_rate:", self.learning_rate)
        print("if_label_normed:", self.if_label_normed)
        print("if_mask_only:", self.if_mask_only)


def main(_):
    with tf.get_default_graph().as_default() as graph:
        # load data
        train_data = np.load(FLAGS.data_dir + FLAGS.train_data)
        valid_data = np.load(FLAGS.data_dir + FLAGS.valid_data)
        # generate raw_data and corrupt_data
        input_train, label_train, _ = utils.generate_input_and_label(
            train_data, FLAGS.aug_ratio, FLAGS.corrupt_ratio, policy='random_data')
        input_valid, label_valid, _ = utils.generate_input_and_label(
            valid_data, FLAGS.aug_ratio, FLAGS.corrupt_ratio, policy='random_data')
        # data normalization
        Norm_er = utils.Norm()
        input_train = Norm_er.data_normalization(input_train)[:, :, :, :6]
        input_valid = Norm_er.data_normalization(input_valid)[:, :, :, :6]
        label_train = label_train[:, :, :, 1:4]
        label_valid = label_valid[:, :, :, 1:4]

        # number of batches
        train_num_batch = input_train.shape[0] // FLAGS.batch_size
        valid_num_batch = input_valid.shape[0] // FLAGS.batch_size
        print(train_num_batch)
        print(valid_num_batch)
        # config setting
        config = TrainingConfig(
            FILTER_NUMBERS, FILTER_STRIDES, input_train.shape)
        config.show()
        # model
        model = model_dcae.DCAEModel(config, graph=graph)
        init = tf.global_variables_initializer()
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        # summary writter
        train_summary_writer = tf.summary.FileWriter(
            FLAGS.log_dir + 'ephoch_train', graph=graph)
        valid_summary_writer = tf.summary.FileWriter(
            FLAGS.log_dir + 'ephoch_valid', graph=graph)

        # Session
        with tf.Session() as sess:
            sess.run(init)
            # restore model if exist
            if FLAGS.restore_path is not None:
                saver.restore(sess, FLAGS.restore_path)
                print("Model restored:", FLAGS.restore_path)
            # training
            for _ in range(FLAGS.total_epoches):
                # time cost evaluation
                start_time = time.time()
                # Shuffle the data
                shuffled_indexes = np.random.permutation(input_train.shape[0])
                input_train = input_train[shuffled_indexes]
                label_train = label_train[shuffled_indexes]
                train_loss_sum = 0.0
                train_sep_loss_sum = np.zeros(shape=[3], dtype=np.float)
                for b in range(train_num_batch):
                    batch_idx = b * FLAGS.batch_size
                    # input, label
                    input_train_batch = input_train[batch_idx:batch_idx +
                                                    FLAGS.batch_size]
                    label_train_batch = label_train[batch_idx:batch_idx +
                                                    FLAGS.batch_size]
                    # train one batch
                    losses, sep_loss, global_step = model.step(
                        sess, input_train_batch, label_train_batch)
                    train_loss_sum += losses
                    train_sep_loss_sum += sep_loss
                global_ephoch = int(global_step // train_num_batch)

                # validation
                valid_loss_sum = 0.0
                valid_sep_loss_sum = np.zeros(shape=[3], dtype=np.float)
                for valid_b in range(valid_num_batch):
                    batch_idx = valid_b * FLAGS.batch_size
                    # input, label
                    input_valid_batch = input_valid[batch_idx:batch_idx +
                                                    FLAGS.batch_size]
                    label_valid_batch = label_valid[batch_idx:batch_idx +
                                                    FLAGS.batch_size]
                    valid_losses, sep_loss = model.compute_loss(
                        sess, input_valid_batch, label_valid_batch)
                    valid_loss_sum += valid_losses
                    valid_sep_loss_sum += sep_loss
                end_time = time.time()

                # logging per ephoch
                print("%d epoches, %d steps, mean train loss: %f, valid mean loss: %f, step cost: %f(sec)" %
                      (global_ephoch,
                       global_step,
                       train_loss_sum / train_num_batch,
                       valid_loss_sum / valid_num_batch,
                       (end_time - start_time)))
                print("train, %f density_loss, %f flow_loss, %f speed_loss" %
                      (train_sep_loss_sum[0] / train_num_batch,
                       train_sep_loss_sum[1] / train_num_batch,
                       train_sep_loss_sum[2] / train_num_batch))
                
                print("valid, %f density_loss, %f flow_loss, %f speed_loss\n" %
                      (valid_sep_loss_sum[0] / valid_num_batch,
                       valid_sep_loss_sum[1] / valid_num_batch,
                       valid_sep_loss_sum[2] / valid_num_batch))

                # train mean ephoch loss
                train_scalar_summary = tf.Summary()
                train_scalar_summary.value.add(
                    simple_value=train_loss_sum / train_num_batch, tag="mean loss")
                train_summary_writer.add_summary(
                    train_scalar_summary, global_step=global_step)
                train_summary_writer.flush()
                # valid mean ephoch loss
                valid_scalar_summary = tf.Summary()
                valid_scalar_summary.value.add(
                    simple_value=valid_loss_sum / valid_num_batch, tag="mean loss")
                valid_summary_writer.add_summary(
                    valid_scalar_summary, global_step=global_step)
                valid_summary_writer.flush()

                # save checkpoints
                if (global_ephoch % FLAGS.save_freq) == 0:
                    save_path = saver.save(
                        sess, FLAGS.checkpoints_dir + "model.ckpt",
                        global_step=global_step)
                    print("Model saved in file: %s" % save_path)


if __name__ == "__main__":
    if not os.path.exists(FLAGS.checkpoints_dir):
        os.makedirs(FLAGS.checkpoints_dir)
    tf.app.run()
