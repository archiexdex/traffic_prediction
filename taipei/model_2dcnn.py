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

        self.global_step = tf.train.get_or_create_global_step(graph=graph)
        self.X_ph = tf.placeholder(dtype=tf.float32, shape=[
            None, 12, 34, 5], name='input_data')
        self.Y_ph = tf.placeholder(dtype=tf.float32, shape=[
            None, 34], name='label_data')
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
        print("inputs:", inputs)
        with tf.variable_scope('conv1') as scope:
            kernel_init = tf.truncated_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            bias_init = tf.random_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            conv1 = tf.layers.conv2d(inputs=inputs, filters=64, kernel_size=[3, 3],
                                     strides=1, padding='valid', activation=tf.nn.relu,
                                     kernel_initializer=kernel_init, bias_initializer=bias_init,
                                     name=scope.name, reuse=scope.reuse)
            print("conv1:", conv1)

        with tf.variable_scope('conv2') as scope:
            kernel_init = tf.truncated_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            bias_init = tf.random_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            conv2 = tf.layers.conv2d(inputs=conv1, filters=128, kernel_size=[3, 3],
                                     strides=1, padding='valid', activation=tf.nn.relu,
                                     kernel_initializer=kernel_init, bias_initializer=bias_init,
                                     name=scope.name, reuse=scope.reuse)
            print("conv2:", conv2)

        with tf.variable_scope('conv3') as scope:
            kernel_init = tf.truncated_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            bias_init = tf.random_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            conv3 = tf.layers.conv2d(inputs=conv2, filters=128, kernel_size=[3, 3],
                                     strides=1, padding='valid', activation=tf.nn.relu,
                                     kernel_initializer=kernel_init, bias_initializer=bias_init,
                                     name=scope.name, reuse=scope.reuse)
            print("conv3:", conv3)

        with tf.variable_scope('conv4') as scope:
            kernel_init = tf.truncated_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            bias_init = tf.random_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            conv4 = tf.layers.conv2d(inputs=conv3, filters=34, kernel_size=[6, 28],
                                     strides=1, padding='valid', activation=tf.nn.relu,
                                     kernel_initializer=kernel_init, bias_initializer=bias_init,
                                     name=scope.name, reuse=scope.reuse)
            print("conv4:", conv4)

        with tf.variable_scope('reshape') as scope:
            reshaped = tf.reshape(
                conv4, [-1, 34], name=scope.name)
            print("reshape:", reshaped)

        return reshaped


    def losses(self, logits, labels):
        """
        Param:
            logits:
            labels: placeholder, shape=[None, 34]
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
        train_op = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(loss,
                                                       global_step=global_step)
        # train_op = tf.train.RMSPropOptimizer(
        #     self.learning_rate, decay=0.99, momentum=0.9, epsilon=1e-10).minimize(loss, global_step=global_step)
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


def test():
    with tf.Graph().as_default() as g:
        model = TFPModel(TestingConfig(), graph=g)
        # train
        # model.step()


if __name__ == "__main__":
    test()
