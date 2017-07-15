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
        self.learning_rate = config.learning_rate
        self.decay_rate = config.decay_rate
        self.momentum = config.momentum

        self.global_step = tf.train.get_or_create_global_step(graph=graph)
        self.X_ph = tf.placeholder(dtype=tf.float32, shape=[
            None, 128, 128, 30, 5], name='input_data')
        self.Y_ph = tf.placeholder(dtype=tf.float32, shape=[
            None, 128], name='input_data')
        self.logits = self.inference(self.X_ph)
        self.losses = self.losses(self.logits, self.Y_ph)
        self.train_op = self.train(self.losses, self.global_step)

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
                                       strides=1, padding='SAME', activation=tf.nn.relu,
                                       kernel_initializer=kernel_init, bias_initializer=bias_init,
                                       name=scope.name, reuse=scope.reuse)
            self._activation_summary(conv1_1)
            print("conv1_1:", conv1_1.shape)

        with tf.variable_scope('conv2_1') as scope:
            kernel_init = tf.truncated_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            bias_init = tf.random_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            conv2_1 = tf.layers.conv3d(inputs=conv1_1, filters=64, kernel_size=3,
                                       strides=1, padding='SAME', activation=tf.nn.relu,
                                       kernel_initializer=kernel_init, bias_initializer=bias_init,
                                       name=scope.name, reuse=scope.reuse)
            self._activation_summary(conv2_1)
            print("conv2_1:", conv2_1.shape)

        with tf.variable_scope('conv3_1') as scope:
            kernel_init = tf.truncated_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            bias_init = tf.random_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            conv3_1 = tf.layers.conv3d(inputs=conv2_1, filters=128, kernel_size=3,
                                       strides=1, padding='SAME', activation=tf.nn.relu,
                                       kernel_initializer=kernel_init, bias_initializer=bias_init,
                                       name=scope.name, reuse=scope.reuse)
            self._activation_summary(conv3_1)
            print("conv3_1:", conv3_1.shape)
        with tf.variable_scope('conv3_2') as scope:
            kernel_init = tf.truncated_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            bias_init = tf.random_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            conv3_2 = tf.layers.conv3d(inputs=conv3_1, filters=128, kernel_size=3,
                                       strides=1, padding='SAME', activation=tf.nn.relu,
                                       kernel_initializer=kernel_init, bias_initializer=bias_init,
                                       name=scope.name, reuse=scope.reuse)
            self._activation_summary(conv3_2)
            print("conv3_2:", conv3_2.shape)

        with tf.variable_scope('conv4') as scope:
            kernel_init = tf.truncated_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            bias_init = tf.random_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            conv4 = tf.layers.conv3d(inputs=conv3_2, filters=1, kernel_size=3,
                                       strides=1, padding='SAME', activation=tf.nn.relu,
                                       kernel_initializer=kernel_init, bias_initializer=bias_init,
                                       name=scope.name, reuse=scope.reuse)
            self._activation_summary(conv4)
            print("conv4:", conv4.shape)

        with tf.variable_scope('final') as scope:
            reshaped_conv4 = tf.reshape(conv4,[-1,128,128,30])
            kernel_init = tf.truncated_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            bias_init = tf.random_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            final = tf.layers.conv2d(inputs=reshaped_conv4, filters=1, kernel_size=3,
                                       strides=1, padding='SAME', activation=tf.nn.relu,
                                       kernel_initializer=kernel_init, bias_initializer=bias_init,
                                       name=scope.name, reuse=scope.reuse)
            self._activation_summary(final)
            print("final:", final.shape)

        return final

        # with tf.variable_scope('conv4_1') as scope:
        #     kernel_init = tf.truncated_normal_initializer(
        #         mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
        #     bias_init = tf.random_normal_initializer(
        #         mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
        #     conv4_1 = tf.layers.conv3d(inputs=max_pool3, filters=256, kernel_size=3,
        #                                strides=1, padding='SAME', activation=tf.nn.relu,
        #                                kernel_initializer=kernel_init, bias_initializer=bias_init,
        #                                name=scope.name, reuse=scope.reuse)
        #     self._activation_summary(conv4_1)
        #     print("conv4_1:", conv4_1.shape)
        # with tf.variable_scope('conv4_2') as scope:
        #     kernel_init = tf.truncated_normal_initializer(
        #         mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
        #     bias_init = tf.random_normal_initializer(
        #         mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
        #     conv4_2 = tf.layers.conv3d(inputs=conv4_1, filters=256, kernel_size=3,
        #                                strides=1, padding='SAME', activation=tf.nn.relu,
        #                                kernel_initializer=kernel_init, bias_initializer=bias_init,
        #                                name=scope.name, reuse=scope.reuse)
        #     self._activation_summary(conv4_2)
        #     print("conv4_2:", conv4_2.shape)
        # with tf.variable_scope('max_pool4') as scope:
        #     max_pool4 = tf.layers.max_pooling3d(
        #         inputs=conv4_2,
        #         pool_size=[3, 3, 3],
        #         strides=[2, 2, 2],
        #         padding='SAME',
        #         data_format='channels_last',
        #         name=scope.name
        #     )
        #     print("max_pool4:", max_pool4.shape)

        # with tf.variable_scope('conv5_1') as scope:
        #     kernel_init = tf.truncated_normal_initializer(
        #         mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
        #     bias_init = tf.random_normal_initializer(
        #         mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
        #     conv5_1 = tf.layers.conv3d(inputs=max_pool4, filters=256, kernel_size=3,
        #                                strides=1, padding='SAME', activation=tf.nn.relu,
        #                                kernel_initializer=kernel_init, bias_initializer=bias_init,
        #                                name=scope.name, reuse=scope.reuse)
        #     self._activation_summary(conv5_1)
        #     print("conv5_1:", conv5_1.shape)
        # with tf.variable_scope('conv5_2') as scope:
        #     kernel_init = tf.truncated_normal_initializer(
        #         mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
        #     bias_init = tf.random_normal_initializer(
        #         mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
        #     conv5_2 = tf.layers.conv3d(inputs=conv5_1, filters=256, kernel_size=3,
        #                                strides=1, padding='SAME', activation=tf.nn.relu,
        #                                kernel_initializer=kernel_init, bias_initializer=bias_init,
        #                                name=scope.name, reuse=scope.reuse)
        #     self._activation_summary(conv5_2)
        #     print("conv5_2:", conv5_2.shape)
        # with tf.variable_scope('max_pool5') as scope:
        #     max_pool5 = tf.layers.max_pooling3d(
        #         inputs=conv5_2,
        #         pool_size=[3, 3, 3],
        #         strides=[2, 2, 2],
        #         padding='SAME',
        #         data_format='channels_last',
        #         name=scope.name
        #     )
        #     print("max_pool5:", max_pool5.shape)

        # with tf.variable_scope('conv6') as scope:
        #     kernel_init = tf.truncated_normal_initializer(
        #         mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
        #     bias_init = tf.random_normal_initializer(
        #         mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
        #     conv6 = tf.layers.conv3d(inputs=max_pool5, filters=1, kernel_size=[4,4,1],
        #                                strides=1, padding='SAME', activation=tf.nn.relu,
        #                                kernel_initializer=kernel_init, bias_initializer=bias_init,
        #                                name=scope.name, reuse=scope.reuse)
        #     self._activation_summary(conv5_2)
        #     print("conv6:", conv6.shape)
        # return conv6

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
            self.learning_rate, self.decay_rate, self.momentum,
            1e-10).minimize(loss, global_step=global_step)
        return train_op

    def step(self, sess, inputs, labels):
        feed_dict = {self.X_ph: inputs,
                     self.Y_ph: labels}
        _, losses = sess.run([self.train_op, self.losses], feed_dict=feed_dict)
        return losses

    # def l2_losses(self, logits, labels):
    #     """
    #     Param:
    #         logits:
    #         labels:
    #     """
    #     with tf.name_scope('squared_difference'):
    #         losses = tf.squared_difference(logits, labels)
    #     return losses

    # def MAPE(self, logits, labels):
    #     """
    #     Param:
    #         logits:
    #         labels:
    #     """
    #     with tf.name_scope('MAPE'):
    #         diff = tf.abs(tf.subtract(logits, labels))
    #         con_less = tf.less(labels, 1)
    #         norn_less = tf.divide(diff, 1)
    #         norn_normal = tf.divide(diff, labels)
    #         norn = tf.where(con_less, norn_less, norn_normal)
    #         mape = tf.reduce_mean(norn)
    #     tf.summary.scalar('MAPE', mape)
    #     return mape

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
        self.batch_size = 256
        self.total_epoches = 10
        self.learning_rate = 0.001
        self.decay_rate = 0.99
        self.momentum = 0.9


def test():
    with tf.Graph().as_default() as g:
        model = TFPModel(TestingConfig(), graph=g)
        # train
        # model.step()

if __name__ == "__main__":
    test()
