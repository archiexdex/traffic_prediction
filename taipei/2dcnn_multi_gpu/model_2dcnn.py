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
        self.num_gpus = config.num_gpus

        self.global_step = tf.train.get_or_create_global_step(graph=graph)
        self.X_ph = tf.placeholder(dtype=tf.float32, shape=[
            None, 70, 12, 5], name='input_data')
        self.Y_ph = tf.placeholder(dtype=tf.float32, shape=[
            None, 35], name='label_data')

        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate)

        tower_grads = []
        num_batch_per_gpu = self.batch_size // self.num_gpus
        with tf.variable_scope(tf.get_variable_scope()):  # get current variable scope
            for i in range(self.num_gpus):
                batch_idx = i * num_batch_per_gpu
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('tower_%d' % i) as scope:
                        logits = self.inference(
                            self.X_ph[batch_idx:batch_idx + num_batch_per_gpu])
                        self.losses = self.loss_function(
                            logits, self.Y_ph[batch_idx:batch_idx + num_batch_per_gpu])
                        # tf.summary.scalar('loss', losses)
                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()
                        grads = optimizer.compute_gradients(self.losses)
                        tower_grads.append(grads)
        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = self.average_gradients(tower_grads)
        self.train_op = optimizer.apply_gradients(
            grads, global_step=self.global_step)

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
                                     strides=1, padding='same', activation=tf.nn.relu,
                                     kernel_initializer=kernel_init, bias_initializer=bias_init,
                                     name=scope.name, reuse=scope.reuse)
            print("conv1:", conv1)

        with tf.variable_scope('conv1_2') as scope:
            kernel_init = tf.truncated_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            bias_init = tf.random_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            conv1_2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=[3, 3],
                                       strides=1, padding='same', activation=tf.nn.relu,
                                       kernel_initializer=kernel_init, bias_initializer=bias_init,
                                       name=scope.name, reuse=scope.reuse)
            print("conv1_2:", conv1_2)

        with tf.variable_scope('conv2') as scope:
            kernel_init = tf.truncated_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            bias_init = tf.random_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            conv2 = tf.layers.conv2d(inputs=conv1_2, filters=128, kernel_size=[3, 3],
                                     strides=2, padding='same', activation=tf.nn.relu,
                                     kernel_initializer=kernel_init, bias_initializer=bias_init,
                                     name=scope.name, reuse=scope.reuse)
            print("conv2:", conv2)

        with tf.variable_scope('conv3') as scope:
            kernel_init = tf.truncated_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            bias_init = tf.random_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            conv3 = tf.layers.conv2d(inputs=conv2, filters=128, kernel_size=[3, 3],
                                     strides=1, padding='same', activation=tf.nn.relu,
                                     kernel_initializer=kernel_init, bias_initializer=bias_init,
                                     name=scope.name, reuse=scope.reuse)
            print("conv3:", conv3)

        with tf.variable_scope('conv4') as scope:
            kernel_init = tf.truncated_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            bias_init = tf.random_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            conv4 = tf.layers.conv2d(inputs=conv3, filters=256, kernel_size=[3, 3],
                                     strides=2, padding='same', activation=tf.nn.relu,
                                     kernel_initializer=kernel_init, bias_initializer=bias_init,
                                     name=scope.name, reuse=scope.reuse)
            print("conv4:", conv4)

        with tf.variable_scope('conv4_2') as scope:
            kernel_init = tf.truncated_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            bias_init = tf.random_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            conv4_2 = tf.layers.conv2d(inputs=conv4, filters=256, kernel_size=[3, 3],
                                       strides=1, padding='same', activation=tf.nn.relu,
                                       kernel_initializer=kernel_init, bias_initializer=bias_init,
                                       name=scope.name, reuse=scope.reuse)
            print("conv4_2:", conv4_2)

        with tf.variable_scope('fully1') as scope:
            kernel_init = tf.truncated_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            bias_init = tf.random_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            flat = tf.contrib.layers.flatten(inputs=conv4_2)
            fully1 = tf.contrib.layers.fully_connected(flat,
                                                       1024,
                                                       activation_fn=tf.nn.relu,
                                                       weights_initializer=kernel_init,
                                                       biases_initializer=bias_init,
                                                       scope=scope, reuse=scope.reuse)
            print("fully1:", fully1)

        with tf.variable_scope('fully2') as scope:
            kernel_init = tf.truncated_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            bias_init = tf.random_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            fully2 = tf.contrib.layers.fully_connected(fully1,
                                                       35,
                                                       activation_fn=tf.nn.relu,
                                                       weights_initializer=kernel_init,
                                                       biases_initializer=bias_init,
                                                       scope=scope, reuse=scope.reuse)
            print("fully2:", fully2)

        return fully2

    def loss_function(self, logits, labels):
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

    def step(self, sess, inputs, labels):
        feed_dict = {self.X_ph: inputs,
                     self.Y_ph: labels}
        losses, global_steps, _ = sess.run(
            [self.losses, self.global_step, self.train_op], feed_dict=feed_dict)
        # summary testing loss
        # self.train_summary_writer.add_summary(
        #     summary, global_step=global_steps)
        return losses, global_steps

    def compute_loss(self, sess, inputs, labels):
        feed_dict = {self.X_ph: inputs,
                     self.Y_ph: labels}
        losses = sess.run(self.losses, feed_dict=feed_dict)
        return losses

    def average_gradients(self, tower_grads):
        """
        Calculate the average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.
        Args:
            tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
            List of pairs of (gradient, variable) where the gradient has been averaged
            across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            # print(grad_and_vars)
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over
                # below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads


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
        self.num_gpus = 2


def test():
    with tf.Graph().as_default() as g:
        model = TFPModel(TestingConfig(), graph=g)
        # train
        X = np.zeros(shape=[256, 70, 12, 5])
        Y = np.zeros(shape=[256, 35])
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            model.step(sess, X, Y)


if __name__ == "__main__":
    test()
