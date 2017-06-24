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
            config:
            is_training:
        """
        self.is_training = is_training
        self.batch_size = config.batch_size
        self.vd_amount = config.vd_amount
        self.total_interval = config.total_interval
        self.learning_rate = config.learning_rate
        self.decay_rate = config.decay_rate
        self.momentum = config.momentum

    def inference(self, inputs):
        """
        Param:
        """
        print(inputs)
        with tf.variable_scope('conv1') as scope:
            kernel_init = tf.truncated_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            bias_init = tf.random_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            conv1 = tf.layers.conv2d(inputs=inputs, filters=64, kernel_size=3,
                                     strides=1, padding='valid', activation=tf.nn.relu,
                                     kernel_initializer=kernel_init, bias_initializer=bias_init,
                                     name=scope.name, reuse=scope.reuse)
            self._activation_summary(conv1)
            print("conv1:", conv1)

        with tf.variable_scope('max_pool') as scope:
            max_pool = tf.layers.max_pooling2d(
                inputs=conv1,
                pool_size=2,
                strides=2,
                padding='valid',
                data_format='channels_last',
                name=scope.name
            )
            print("max_pool:", max_pool)

        with tf.variable_scope('conv2') as scope:
            kernel_init = tf.truncated_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            bias_init = tf.random_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            conv2 = tf.layers.conv2d(inputs=max_pool, filters=128, kernel_size=3,
                                     strides=1, padding='valid', activation=tf.nn.relu,
                                     kernel_initializer=kernel_init, bias_initializer=bias_init,
                                     name=scope.name, reuse=scope.reuse)
            self._activation_summary(conv2)
            print("conv2:", conv2)

        with tf.variable_scope('conv3') as scope:
            kernel_init = tf.truncated_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            bias_init = tf.random_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            conv3 = tf.layers.conv2d(inputs=conv2, filters=128, kernel_size=3,
                                     strides=1, padding='valid', activation=tf.nn.relu,
                                     kernel_initializer=kernel_init, bias_initializer=bias_init,
                                     name=scope.name, reuse=scope.reuse)
            self._activation_summary(conv3)
            print("conv3:", conv3)

        with tf.variable_scope('full') as scope:
            kernel_init = tf.truncated_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            bias_init = tf.random_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            full = tf.layers.conv2d(inputs=conv3, filters=28, kernel_size=[1,9],
                                     strides=1, padding='valid', activation=tf.nn.relu,
                                     kernel_initializer=kernel_init, bias_initializer=bias_init,
                                     name=scope.name, reuse=scope.reuse)
            self._activation_summary(full)
            print("full:", full)

        with tf.variable_scope('reshape') as scope:
            reshaped = tf.reshape(
                full, [-1, 28], name=scope.name)
            print("reshape:", reshaped)

        return reshaped

    def losses(self, logits, labels):
        """
        Param:
            logits:
            labels:
        """
        with tf.name_scope('l2_loss'):
            losses = tf.squared_difference(logits, labels)
            l2_loss = tf.reduce_mean(losses)
        tf.summary.scalar('l2_loss', l2_loss)
        return l2_loss

    def l2_losses(self, logits, labels):
        """
        Param:
            logits:
            labels:
        """
        with tf.name_scope('squared_difference'):
            losses = tf.squared_difference(logits, labels)
        return losses

    def MAPE(self, logits, labels):
        """
        Param:
            logits:
            labels:
        """
        with tf.name_scope('MAPE'):
            diff = tf.abs(tf.subtract(logits, labels))
            con_less = tf.less(labels, 1)
            norn_less = tf.divide(diff, 1)
            norn_normal = tf.divide(diff, labels)
            norn = tf.where(con_less, norn_less, norn_normal)
            mape = tf.reduce_mean(norn)
        tf.summary.scalar('MAPE', mape)
        return mape

    def train(self, loss, global_step=None):
        """
        Param:
            loss:
        """
        # train_op = tf.train.AdamOptimizer(
        #     learning_rate=self.learning_rate).minimize(loss,
        #                                                global_step=global_step)
        train_op = tf.train.RMSPropOptimizer(
            self.learning_rate, self.decay_rate, self.momentum,
            1e-10).minimize(loss, global_step=global_step)
        return train_op

    def _activation_summary(self, x):
        """Helper to create summaries for activations.
        Creates a summary that provides a histogram of activations.
        Creates a summary that measures the sparsity of activations.
        Args:
        x: Tensor
        Returns: nothing
        """
        tensor_name = x.op.name
        tf.summary.histogram(tensor_name + '/activations', x)
        tf.summary.scalar(tensor_name + '/sparsity',
                        tf.nn.zero_fraction(x))


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
        self.decay_rate = 0.99
        self.momentum = 0.9

def test():
    X_ph = tf.placeholder(dtype=tf.float32, shape=[
                            512, 12, 28, 5], name='input_data')
    model = TFPModel(TestingConfig())
    model.inference(X_ph)

if __name__ == "__main__":
    test()
