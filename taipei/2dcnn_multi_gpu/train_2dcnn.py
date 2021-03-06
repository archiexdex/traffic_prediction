from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import tensorflow as tf
import model_2dcnn
import parameter_saver

FLAGS = tf.app.flags.FLAGS

# path parameters
tf.app.flags.DEFINE_string("train_data", "train_data.npy",
                           "training data name")
tf.app.flags.DEFINE_string("test_data", "test_data.npy",
                           "testing data name")
tf.app.flags.DEFINE_string("train_label", "train_label.npy",
                           "training label data name")
tf.app.flags.DEFINE_string("test_label", "test_label.npy",
                           "testing label data name")
tf.app.flags.DEFINE_string('data_dir', '/home/xdex/Desktop/traffic_flow_detection/taipei/training_data/new_raw_data/vd_base/',
                           "data directory")
tf.app.flags.DEFINE_string('checkpoints_dir', 'v3/checkpoints/',
                           "training checkpoints directory")
tf.app.flags.DEFINE_string('log_dir', 'v3/log/',
                           "summary directory")
# training parameters
tf.app.flags.DEFINE_integer('batch_size', 512,
                            "mini-batch size")
tf.app.flags.DEFINE_integer('total_epoches', 5000,
                            "total training epoches")
tf.app.flags.DEFINE_integer('save_freq', 25,
                            "number of epoches to saving model")
tf.app.flags.DEFINE_integer('total_interval', 12,
                            "total steps of time")
tf.app.flags.DEFINE_float('learning_rate', 0.00001,
                          "learning rate of AdamOptimizer")
tf.app.flags.DEFINE_integer('num_gpus', 2,
                            "multi gpu")
tf.app.flags.DEFINE_string('restore_path', None,
                           "path of saving model eg: checkpoints/model.ckpt-5")


class ModelConfig(object):
    """
    testing config
    """

    def __init__(self, train_shape, test_shape):
        self.data_dir = FLAGS.data_dir
        self.checkpoints_dir = FLAGS.checkpoints_dir
        self.log_dir = FLAGS.log_dir
        self.batch_size = FLAGS.batch_size
        self.total_epoches = FLAGS.total_epoches
        self.save_freq = FLAGS.save_freq
        self.total_interval = FLAGS.total_interval
        self.learning_rate = FLAGS.learning_rate
        self.num_gpus = FLAGS.num_gpus
        self.train_shape = train_shape
        self.test_shape = test_shape
        self.is_test = False

    def show(self):
        print("data_dir:", self.data_dir)
        print("checkpoints_dir:", self.checkpoints_dir)
        print("log_dir:", self.log_dir)
        print("batch_size:", self.batch_size)
        print("total_epoches:", self.total_epoches)
        print("save_freq:", self.save_freq)
        print("total_interval:", self.total_interval)
        print("learning_rate:", self.learning_rate)
        print("num_gpus:", self.num_gpus)
        print("train_shape:", self.train_shape)
        print("test_shape:", self.test_shape)
        print("is_test:", self.is_test)

def main(_):
    with tf.get_default_graph().as_default() as graph:
        # training loss saver
        loss_saver = parameter_saver.Training_loss_saver()
        loss_saver.add_parameter(
            "description", "remove some bad VD train(VPJFZ00), label(VP8GI20, VP8GI60, VP8GX40, VMXH820)")
        # load data
        train_data = np.load(FLAGS.data_dir + FLAGS.train_data)[:, :, :, :]
        test_data = np.load(FLAGS.data_dir + FLAGS.test_data)[:, :, :, :]
        train_label = np.load(FLAGS.data_dir + FLAGS.train_label)
        test_label = np.load(FLAGS.data_dir + FLAGS.test_label)
        # number of batches
        train_num_batch = train_data.shape[0] // FLAGS.batch_size
        test_num_batch = test_data.shape[0] // FLAGS.batch_size
        print(train_num_batch)
        print(test_num_batch)
        loss_saver.add_parameter("input_shape", train_data.shape)
        loss_saver.add_parameter("test_shape", train_label.shape)
        # config setting
        config = ModelConfig(train_data.shape, train_label.shape)
        config.show()
        # model
        model = model_2dcnn.TFPModel(config, graph=graph)
        # Add an op to initialize the variables.
        init = tf.global_variables_initializer()
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        # summary writter
        train_summary_writer = tf.summary.FileWriter(
            FLAGS.log_dir + 'ephoch_train', graph=graph)
        valid_summary_writer = tf.summary.FileWriter(
            FLAGS.log_dir + 'ephoch_valid', graph=graph)

        # Session
        with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
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
                shuffled_indexes = np.random.permutation(train_data.shape[0])
                train_data = train_data[shuffled_indexes]
                train_label = train_label[shuffled_indexes]
                each_vd_losses_sum = []
                train_loss_sum = 0.0
                for b in range(train_num_batch):
                    batch_idx = b * FLAGS.batch_size
                    # input, label
                    train_data_batch = train_data[batch_idx:batch_idx +
                                                  FLAGS.batch_size]
                    train_label_batch = train_label[batch_idx:batch_idx +
                                                    FLAGS.batch_size]
                    # train
                    each_vd_losses, losses, global_step = model.step(
                        sess, train_data_batch, train_label_batch)
                    each_vd_losses_sum.append(each_vd_losses)
                    train_loss_sum += losses
                each_vd_losses_sum = np.array(each_vd_losses_sum)
                each_vd_losses_mean = np.mean(each_vd_losses_sum, axis=0)
                global_ephoch = int(global_step // train_num_batch)

                # validation
                test_each_vd_losses_sum = []
                test_loss_sum = 0.0
                for test_b in range(test_num_batch):
                    batch_idx = test_b * FLAGS.batch_size
                    # input, label
                    test_data_batch = test_data[batch_idx:batch_idx +
                                                FLAGS.batch_size]
                    test_label_batch = test_label[batch_idx:batch_idx +
                                                  FLAGS.batch_size]
                    test_each_vd_losses, test_losses = model.compute_loss(
                        sess, test_data_batch, test_label_batch)
                    test_each_vd_losses_sum.append(test_each_vd_losses)
                    test_loss_sum += test_losses
                test_each_vd_losses_sum = np.array(test_each_vd_losses_sum)
                test_each_vd_losses_mean = np.mean(
                    test_each_vd_losses_sum, axis=0)
                end_time = time.time()
                # logging per ephoch
                print("%d epoches, %d steps, mean train loss: %f, test mean loss: %f, time cost: %f(sec/batch)" %
                      (global_ephoch,
                       global_step,
                       train_loss_sum / train_num_batch,
                       test_loss_sum / test_num_batch,
                       (end_time - start_time) / train_num_batch))
                print("each train vd's mean loss:")
                print(each_vd_losses_mean)
                print("each test vd's mean loss:")
                print(test_each_vd_losses_mean)

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
                    simple_value=test_loss_sum / test_num_batch, tag="mean loss")
                valid_summary_writer.add_summary(
                    valid_scalar_summary, global_step=global_step)
                valid_summary_writer.flush()

                # save checkpoints
                if (global_ephoch % FLAGS.save_freq) == 0:
                    save_path = saver.save(
                        sess, FLAGS.checkpoints_dir + "model.ckpt",
                        global_step=global_step)
                    print("Model saved in file: %s" % save_path)

                # save the loss
                loss_saver.add_loss(global_ephoch,
                                    train_loss_sum / train_num_batch,
                                    test_loss_sum / test_num_batch)
                loss_saver.save()


if __name__ == "__main__":
    if not os.path.exists(FLAGS.checkpoints_dir):
        os.makedirs(FLAGS.checkpoints_dir)
    tf.app.run()
