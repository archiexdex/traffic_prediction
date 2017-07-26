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

    def __init__(self, config, is_training=True):
        """
        Param:
            config: the whole hyper perameters
        """
        self.batch_size = config.batch_size
        self.vd_amount = config.vd_amount
        self.total_interval = config.total_interval
        if is_training:
            self.learning_rate = config.learning_rate
        else:
            self.learning_rate = 0.0

    def inference(self, inputs):
        """
        Param:
            inputs: [batch_size, times, vds, features]
        """
        print("inputs:", inputs)
        with tf.variable_scope('conv1') as scope:
            kernel_init = tf.truncated_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            bias_init = tf.random_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            conv1 = tf.layers.conv2d(inputs=inputs, filters=64, kernel_size=[3, 5],
                                     strides=1, padding='valid', activation=tf.nn.relu,
                                     kernel_initializer=kernel_init, bias_initializer=bias_init,
                                     name=scope.name, reuse=scope.reuse)
            print("conv1:", conv1)

        with tf.variable_scope('conv2') as scope:
            kernel_init = tf.truncated_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            bias_init = tf.random_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            conv2 = tf.layers.conv2d(inputs=conv1, filters=128, kernel_size=[3, 5],
                                     strides=1, padding='valid', activation=tf.nn.relu,
                                     kernel_initializer=kernel_init, bias_initializer=bias_init,
                                     name=scope.name, reuse=scope.reuse)
            print("conv2:", conv2)

        with tf.variable_scope('conv3') as scope:
            kernel_init = tf.truncated_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            bias_init = tf.random_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            conv3 = tf.layers.conv2d(inputs=conv2, filters=128, kernel_size=[3, 5],
                                     strides=1, padding='valid', activation=tf.nn.relu,
                                     kernel_initializer=kernel_init, bias_initializer=bias_init,
                                     name=scope.name, reuse=scope.reuse)
            print("conv3:", conv3)

        with tf.variable_scope('conv4') as scope:
            kernel_init = tf.truncated_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            bias_init = tf.random_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            conv4 = tf.layers.conv2d(inputs=conv3, filters=2, kernel_size=[3, 5],
                                     strides=1, padding='valid', activation=tf.nn.relu,
                                     kernel_initializer=kernel_init, bias_initializer=bias_init,
                                     name=scope.name, reuse=scope.reuse)
            print("conv4:", conv4)

        with tf.variable_scope('reshape') as scope:
            reshaped = tf.reshape(
                conv4, [-1, 4, 12, 2], name=scope.name)
            print("reshape:", reshaped)
        return reshaped

    def losses(self, logits, labels):
        """
        Param:
            logits: [batch_size, times, vds, features]
            labels: [batch_size, times, vds, features]
        """
        with tf.name_scope('l2_loss'):
            losses = tf.squared_difference(logits, labels)
            l2_loss = tf.reduce_mean(losses)
        tf.summary.scalar('l2_loss', l2_loss)
        return l2_loss

    def l2_losses(self, logits, labels):
        """
        return the l2 loss for each input.
        Param:
            logits: [batch_size, times, vds, features]
            labels: [batch_size, times, vds, features]
        """
        with tf.name_scope('squared_difference'):
            losses = tf.squared_difference(logits, labels)
        return losses

    def train(self, loss, global_step=None):
        """
        Param:
            loss: scalar
            global_step: training steps
        """
        train_op = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(loss,
                                                       global_step=global_step)
        # train_op = tf.train.RMSPropOptimizer(
        #     learning_rate=self.learning_rate,
        #     decay=0.99,
        #     momentum=0.9,
        #     epsilon=1e-10).minimize(loss, global_step=global_step)
        return train_op


class TestingConfig(object):
    """
    testing config
    """

    def __init__(self):
        self.data_dir = "FLAGS.data_dir"
        self.checkpoints_dir = "FLAGS.checkpoints_dir"
        self.log_dir = "FLAGS.log_dir"
        self.batch_size = 512
        self.total_epoches = 10
        self.vd_amount = 28
        self.total_interval = 12
        self.learning_rate = 0.001


def test():
    """
    testing if the model runable
    """
    X_ph = tf.placeholder(dtype=tf.float32, shape=[
        512, 12, 28, 5], name='input_data')
    model = TFPModel(TestingConfig())
    model.inference(X_ph)


if __name__ == "__main__":
    test()
