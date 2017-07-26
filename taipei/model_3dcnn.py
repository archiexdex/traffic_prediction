from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn


class TFPModel(object):
    """
    The Traffic Flow Prediction Modle
    """

    def __init__(self, config, graph=None):
        """
        Param:
            config:
            graph:
        """
        self.batch_size = config.batch_size
        self.log_dir = config.log_dir
        self.learning_rate = config.learning_rate
        self.row_size = config.row_size
        self.col_size = config.col_size
        self.total_interval = config.total_interval

        self.global_step = tf.train.get_or_create_global_step(graph=graph)
        self.X_ph = tf.placeholder(dtype=tf.float32, shape=[
            None, self.row_size, self.col_size, self.total_interval, 5], name='input_data')
        self.Y_ph = tf.placeholder(dtype=tf.float32, shape=[
            None, 35], name='label_data')
        self.logits = self.inference(self.X_ph)
        self.losses = self.losses(self.logits, self.Y_ph)
        tf.summary.scalar('loss', self.losses)
        self.train_op = self.train(self.losses, self.global_step)
        # summary
        self.merged_op = tf.summary.merge_all()
        # summary writer
        self.train_summary_writer = tf.summary.FileWriter(
            self.log_dir + 'train', graph=graph)

    def inference(self, inputs):
        """
        Param:
        """
        print("inputs:", inputs.shape)
        with tf.variable_scope('conv1_1') as scope:
            kernel_init = tf.truncated_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            bias_init = tf.random_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            conv1_1 = tf.layers.conv3d(inputs=inputs, filters=32, kernel_size=3,
                                       strides=1, padding='VALID', activation=tf.nn.relu,
                                       kernel_initializer=kernel_init, bias_initializer=bias_init,
                                       name=scope.name, reuse=scope.reuse)
            print("conv1_1:", conv1_1.shape)

        with tf.variable_scope('max_pool1') as scope:
            max_pool1 = tf.layers.max_pooling3d(
                inputs=conv1_1,
                pool_size=[3, 3, 1],
                strides=[2, 2, 1],
                padding='SAME',
                data_format='channels_last',
                name=scope.name
            )
            print("max_pool1:", max_pool1.shape)

        with tf.variable_scope('conv2_1') as scope:
            kernel_init = tf.truncated_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            bias_init = tf.random_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            conv2_1 = tf.layers.conv3d(inputs=max_pool1, filters=64, kernel_size=3,
                                       strides=1, padding='VALID', activation=tf.nn.relu,
                                       kernel_initializer=kernel_init, bias_initializer=bias_init,
                                       name=scope.name, reuse=scope.reuse)
            print("conv2_1:", conv2_1.shape)

        with tf.variable_scope('max_pool2') as scope:
            max_pool2 = tf.layers.max_pooling3d(
                inputs=conv2_1,
                pool_size=[3, 3, 1],
                strides=[2, 2, 1],
                padding='SAME',
                data_format='channels_last',
                name=scope.name
            )
            print("max_pool2:", max_pool2.shape)

        with tf.variable_scope('conv3_1') as scope:
            kernel_init = tf.truncated_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            bias_init = tf.random_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            conv3_1 = tf.layers.conv3d(inputs=max_pool2, filters=128, kernel_size=3,
                                       strides=1, padding='VALID', activation=tf.nn.relu,
                                       kernel_initializer=kernel_init, bias_initializer=bias_init,
                                       name=scope.name, reuse=scope.reuse)
            print("conv3_1:", conv3_1.shape)
        with tf.variable_scope('conv3_2') as scope:
            kernel_init = tf.truncated_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            bias_init = tf.random_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            conv3_2 = tf.layers.conv3d(inputs=conv3_1, filters=128, kernel_size=3,
                                       strides=1, padding='VALID', activation=tf.nn.relu,
                                       kernel_initializer=kernel_init, bias_initializer=bias_init,
                                       name=scope.name, reuse=scope.reuse)
            print("conv3_2:", conv3_2.shape)

        with tf.variable_scope('conv4') as scope:
            kernel_init = tf.truncated_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            bias_init = tf.random_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            conv4 = tf.layers.conv3d(inputs=conv3_2, filters=35, kernel_size=[3, 3, 4],
                                     strides=1, padding='VALID', activation=tf.nn.relu,
                                     kernel_initializer=kernel_init, bias_initializer=bias_init,
                                     name=scope.name, reuse=scope.reuse)
            print("conv4:", conv4.shape)

        reshaped_final = tf.reshape(conv4, [-1, 35])
        print("reshaped_final:", reshaped_final.shape)

        return reshaped_final

    def losses(self, logits, labels):
        """
        Param:
            logits:
            labels: placeholder, shape=[None, 35]
        """
        with tf.name_scope('l2_loss'):
            losses = tf.squared_difference(logits, labels)
            l2_loss = tf.reduce_mean(losses)
        print(l2_loss)
        return l2_loss

    def train(self, loss, global_step=None):
        """
        Param:
            loss:
        """
        # train_op = tf.train.AdamOptimizer(
        #     learning_rate=self.learning_rate).minimize(loss,
        #                                                global_step=global_step)
        train_op = tf.train.RMSPropOptimizer(
            self.learning_rate, decay=0.99, momentum=0.9, epsilon=1e-10).minimize(loss, global_step=global_step)
        return train_op

    def step(self, sess, inputs, labels):
        feed_dict = {self.X_ph: inputs,
                     self.Y_ph: labels}
        losses, global_steps, _, summary = sess.run(
            [self.losses, self.global_step, self.train_op, self.merged_op], feed_dict=feed_dict)
        # summary testing loss
        self.train_summary_writer.add_summary(
            summary, global_step=global_steps)
        return losses, global_steps

    def compute_loss(self, sess, inputs, labels):
        feed_dict = {self.X_ph: inputs,
                     self.Y_ph: labels}
        losses = sess.run(self.losses, feed_dict=feed_dict)
        return losses


class TestingConfig(object):
    """
    testing config
    """

    def __init__(self):
        self.data_dir = "FLAGS.data_dir"
        self.checkpoints_dir = "FLAGS.checkpoints_dir"
        self.log_dir = "FLAGS.log_dir"
        self.batch_size = 256
        self.total_epoches = 10
        self.learning_rate = 0.001
        self.decay_rate = 0.99
        self.momentum = 0.9
        self.row_size = 32
        self.col_size = 32
        self.total_interval = 12


def test():
    with tf.Graph().as_default() as g:
        model = TFPModel(TestingConfig(), graph=g)
        # train
        # model.step()


if __name__ == "__main__":
    test()
