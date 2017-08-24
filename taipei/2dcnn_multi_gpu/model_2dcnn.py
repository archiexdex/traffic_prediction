from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import parameter_saver
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
        self.batch_size    = config.batch_size
        self.log_dir       = config.log_dir
        self.learning_rate = config.learning_rate
        self.num_gpus      = config.num_gpus
        self.train_shape   = config.train_shape
        self.test_shape    = config.test_shape
        self.is_test       = config.is_test

        if self.is_test == False:
            self.parameter_saver = parameter_saver.Parameter_saver(
                "2 dimension CNN")
            self.parameter_saver.add_parameter("batch_size", self.batch_size)
            self.parameter_saver.add_parameter("learning_rate", self.learning_rate)
            self.parameter_saver.add_parameter("optimizer", "adam")
            self.parameter_saver.add_parameter("loss_function", "l2_loss")
            self.parameter_saver.add_parameter("log_dir", self.log_dir)
            self.parameter_saver.add_parameter("num_gpu", self.num_gpus)
            self.parameter_saver.add_parameter("train_shape", self.train_shape)
            self.parameter_saver.add_parameter("test_shape", self.test_shape)

        self.global_step = tf.train.get_or_create_global_step(graph=graph)
        self.X_ph = tf.placeholder(dtype=tf.float32, shape=[
            None, self.train_shape[1], self.train_shape[2], self.train_shape[3]], name='input_data')
        self.Y_ph = tf.placeholder(dtype=tf.float32, shape=[
            None, self.test_shape[1]], name='label_data')

        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate)

        vd_losses_sum = []
        tower_grads = []
        num_batch_per_gpu = self.batch_size // self.num_gpus
        with tf.variable_scope(tf.get_variable_scope()):  # get current variable scope
            for i in range(self.num_gpus):
                batch_idx = i * num_batch_per_gpu
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('tower_%d' % i) as scope:
                        self.logits = self.inference(
                            self.X_ph[batch_idx:batch_idx + num_batch_per_gpu])
                        vd_losses, self.losses = self.loss_function(
                            self.logits, self.Y_ph[batch_idx:batch_idx + num_batch_per_gpu])
                        vd_losses_sum.append(vd_losses)
                        # tf.summary.scalar('loss', losses)
                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()
                        grads = optimizer.compute_gradients(self.losses)
                        tower_grads.append(grads)
        # get each vd mean loss of multi gpu
        self.each_vd_losses = tf.reduce_mean(vd_losses_sum, axis=0)
        print(self.each_vd_losses)

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
            inputs: float32, placeholder, shape=[None, num_vds, num_interval, num_features]
        Return:
            fully2: float, shape=[None, num_target_vds]
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
            if self.is_test == False:
                self.parameter_saver.add_layer("conv1", {"filter": 64, "kernel_size": [
                                           3, 3], "stride": 1, "padding": "same", "activation": "relu"})

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
            if self.is_test == False:
                self.parameter_saver.add_layer("conv1_2", {"filter": 64, "kernel_size": [
                                           3, 3], "stride": 1, "padding": "same", "activation": "relu"})

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
            if self.is_test == False:
                self.parameter_saver.add_layer("conv2", {"filter": 128, "kernel_size": [
                                           3, 3], "stride": 2, "padding": "same", "activation": "relu"})

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
            if self.is_test == False:
                self.parameter_saver.add_layer("conv3", {"filter": 128, "kernel_size": [
                                           3, 3], "stride": 1, "padding": "same", "activation": "relu"})

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
            if self.is_test == False:
                self.parameter_saver.add_layer("conv4", {"filter": 256, "kernel_size": [
                                           3, 3], "stride": 2, "padding": "same", "activation": "relu"})

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
            if self.is_test == False:
                self.parameter_saver.add_layer("conv4_2", {"filter": 256, "kernel_size": [
                                           3, 3], "stride": 2, "padding": "same", "activation": "relu"})

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
            if self.is_test == False:
                self.parameter_saver.add_layer(
                "fully1", {"inputs": "flat", "num_outputs": 1024, "activation": "relu"})

        with tf.variable_scope('fully2') as scope:
            kernel_init = tf.truncated_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            bias_init = tf.random_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            fully2 = tf.contrib.layers.fully_connected(fully1,
                                                       self.test_shape[1],
                                                       activation_fn=tf.nn.relu,
                                                       weights_initializer=kernel_init,
                                                       biases_initializer=bias_init,
                                                       scope=scope, reuse=scope.reuse)
            print("fully2:", fully2)
            if self.is_test == False:
                self.parameter_saver.add_layer(
                "fully2", {"inputs": "fully1", "num_outputs": 29, "activation": "relu"})

        if self.is_test == False:
            self.parameter_saver.save()
        return fully2

    def loss_function(self, logits, labels):
        """
        Param:
            logits: float, shape=[None, num_target_vds], inference's output
            labels: float, placeholder, shape=[None, num_target_vds]
        Return:
            vd_losses: float, shape=[num_vds], every vd's loss in one step
            l2_mean_loss: float, shape=[], mean loss for backprop
        """
        with tf.name_scope('l2_loss'):
            vd_losses = tf.squared_difference(logits, labels)
            vd_mean_losses = tf.reduce_mean(vd_losses, axis=0)
            l2_mean_loss = tf.reduce_mean(vd_losses)
        print(l2_mean_loss)
        return vd_mean_losses, l2_mean_loss

    def step(self, sess, inputs, labels):
        """
        Return
            each_vd_losses: float, shape=[num_vds], every vd's loss in one step
            losses: float, shape=[], mean loss for backprop
            global_steps: int, shape=[], training steps
        """
        feed_dict = {self.X_ph: inputs,
                     self.Y_ph: labels}
        each_vd_losses, losses, global_steps, _ = sess.run(
            [self.each_vd_losses, self.losses, self.global_step, self.train_op], feed_dict=feed_dict)
        # summary testing loss
        # self.train_summary_writer.add_summary(
        #     summary, global_step=global_steps)
        return each_vd_losses, losses, global_steps

    def compute_loss(self, sess, inputs, labels):
        feed_dict = {self.X_ph: inputs,
                     self.Y_ph: labels}
        each_vd_losses, losses = sess.run(
            [self.each_vd_losses, self.losses], feed_dict=feed_dict)
        return each_vd_losses, losses

    def predict(self, sess, inputs):
        feed_dict = {self.X_ph: inputs}
        prediction = sess.run(self.logits, feed_dict=feed_dict)
        return prediction

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
        self.data_dir = "FLAGS.data_dir/"
        self.checkpoints_dir = "FLAGS.checkpoints_dir/"
        self.log_dir = "FLAGS.log_dir/"
        self.batch_size = 256
        self.total_epoches = 10
        self.learning_rate = 0.001
        self.num_gpus = 2


def test():
    with tf.Graph().as_default() as g:
        model = TFPModel(TestingConfig(), graph=g)
        # train
        X = np.zeros(shape=[256, 60, 12, 7])
        Y = np.zeros(shape=[256, 7])
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            model.step(sess, X, Y)


if __name__ == "__main__":
    test()
