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
        self.hidden_size = config.hidden_size
        self.rnn_layers = config.rnn_layers
        self.num_steps = config.num_steps
        if config.is_float32:
            self.data_type = tf.float32
        else:
            self.data_type = tf.float16
        self.learning_rate = config.learning_rate
        self.decay_rate = config.decay_rate
        self.momentum = config.momentum

    def inference(self, inputs):
        """
        Param:
        """
        with tf.variable_scope('reshape') as scope:
            reshaped_input = tf.reshape(inputs, [
                                        self.batch_size, self.num_steps, 28], name=scope.name)

        with tf.variable_scope('lstm') as scope:
            cells = rnn.MultiRNNCell(
                [self.lstm_cell() for _ in range(self.rnn_layers)])
            # static method
            lstm_input = tf.unstack(fullycon, num=self.num_steps, axis=1)
            outputs, states = rnn.static_rnn(
                cell=cells, inputs=lstm_input, dtype=tf.float32, scope=scope)

        # with tf.variable_scope('fullycon') as scope:
        #     kernel_init = tf.truncated_normal_initializer(
        #         mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
        #     bias_init = tf.random_normal_initializer(
        #         mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
        #     fullycon = tf.contrib.layers.fully_connected(
        #         inputs=outputs[-1], num_outputs=28, activation_fn=tf.nn.relu,
        #         weights_initializer=kernel_init, biases_initializer=bias_init,
        #         reuse=scope.reuse, trainable=True, scope=scope)

        return outputs[-1]

    def lstm_cell(self):
        return rnn.LSTMCell(self.hidden_size, use_peepholes=True, initializer=None, num_proj=28,
                            forget_bias=1.0, state_is_tuple=True,
                            activation=tf.tanh, reuse=tf.get_variable_scope().reuse)

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

    def MAPE(self, logits, labels):
        """
        Param:
            logits:
            labels:
        """
        with tf.name_scope('MAPE'):
            diff = tf.abs(tf.subtract(logits, labels))
            norn = tf.divide(diff, labels)
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


if __name__ == "__main__":
    pass
