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
            # TODO
            reshaped_input = tf.reshape(inputs, [
                                        self.batch_size, self.num_steps, 28 * 1], name=scope.name)
            print ("reshape:", reshaped_input)

        with tf.variable_scope('lstm') as scope:
            cells = rnn.MultiRNNCell(
                [self.lstm_cell() for _ in range(self.rnn_layers)])

            # dynamic method
            # lstm_input = reshape_back
            # outputs, states = tf.nn.dynamic_rnn(
            #     cell=cells, inputs=lstm_input, dtype=tf.float32, scope=scope)
            # print ("last_logit:", outputs[:, -1, :])

            # static method
            lstm_input = tf.unstack(reshaped_input, num=self.num_steps, axis=1)
            print ("lstm_input", lstm_input)
            outputs, states = rnn.static_rnn(
                cell=cells, inputs=lstm_input, dtype=tf.float32, scope=scope)
            print ("last_logit:", outputs[-1])

            # vanilla method
            # lstm_input = reshape_back
            # state = cell.zero_state(
            #     batch_size=self.batch_size, dtype=tf.float32)
            # logits_list = []
            # for time_step in range(self.num_steps):
            #     if time_step > 0 or not self.is_training:
            #         tf.get_variable_scope().reuse_variables()
            #     cell_out, state = cell(
            #         inputs=lstm_input[:, time_step, :], state=state, scope=scope)
            #     logits_list.append(cell_out)
            # last_logit = logits_list[-1]
            # print ("last_logit:", last_logit)

        return outputs[-1]

    def lstm_cell(self):
        return rnn.LSTMCell(self.hidden_size, use_peepholes=False, initializer=None, num_proj=28,
                            forget_bias=1.0, state_is_tuple=True,
                            activation=tf.tanh, reuse=tf.get_variable_scope().reuse)

    def losses(self, logits, labels):
        """
        Param:
            logits:
            labels:
        """
        # with tf.name_scope('cross_entropy'):
        #     diff = tf.nn.softmax_cross_entropy_with_logits(
        #         labels=labels, logits=logits)
        #     cross_entropy = tf.reduce_mean(diff)
        # return cross_entropy
        losses = tf.squared_difference(logits, labels)
        l2_loss = tf.reduce_mean(losses)
        tf.summary.scalar('l2_loss', l2_loss)
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


if __name__ == "__main__":
    pass
