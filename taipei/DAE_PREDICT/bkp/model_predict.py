from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class TFPModel(object):
    """
    The Traffic Flow Prediction Modle
    """

    def __init__(self, config, input_data, label_data, dae_output, graph=None):
        """
        Param:
            config:
            graph:
        """
        self.batch_size = config.batch_size
        self.log_dir = config.log_dir
        self.learning_rate = config.learning_rate
        self.num_gpus = config.num_gpus
        self.train_shape = config.train_shape
        self.test_shape = config.test_shape

        self.__global_step = tf.train.get_or_create_global_step(graph=graph)
        self.X_ph = input_data
        self.Y_ph = label_data

        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate)

        self.logits = self.inference(dae_output)
        self.each_vd_losses, self.losses = self.loss_function(
            self.logits, self.Y_ph)
        tf.summary.scalar('loss', self.losses)

        self.__train_op = optimizer.minimize(
            self.losses, var_list=self.__get_var_list(), global_step=self.__global_step)

        self.__train_all_op = optimizer.minimize(
            self.losses, global_step=self.__global_step)

        # summary
        self.merged_op = tf.summary.merge_all()
        # summary writer
        self.train_summary_writer = tf.summary.FileWriter(
            self.log_dir + 'PREDICT_train', graph=graph)

    def __get_var_list(self):
        """ 
        """
        trainable_V = tf.trainable_variables()
        theta_PREDICT = []
        for _, v in enumerate(trainable_V):
            if v.name.startswith('PREDICT'):
                theta_PREDICT.append(v)
        return theta_PREDICT

    def inference(self, inputs):
        """
        Param:
            inputs: float32, placeholder, shape=[None, num_vds, num_interval, num_features]
        Return:
            fully2: float, shape=[None, num_target_vds]
        """
        print("inputs:", inputs)
        with tf.variable_scope('PREDICT'):
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
                                                           self.test_shape[1] *
                                                           self.test_shape[2],
                                                           activation_fn=tf.nn.relu,
                                                           weights_initializer=kernel_init,
                                                           biases_initializer=bias_init,
                                                           scope=scope, reuse=scope.reuse)
                print("fully2:", fully2)

            with tf.variable_scope('reshape') as scope:
                reshaped = tf.reshape(
                    fully2, [-1, 18, 4], name=scope.name)
                print("reshape:", reshaped)

        return reshaped

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

    def step(self, sess, inputs, labels, if_train_all):
        """
        Return
            each_vd_losses: float, shape=[num_vds], every vd's loss in one step
            losses: float, shape=[], mean loss for backprop
            global_steps: int, shape=[], training steps
        """
        feed_dict = {self.X_ph: inputs,
                     self.Y_ph: labels}
        if if_train_all:
            train_op = self.__train_all_op
        else:
            train_op = self.__train_op
        each_vd_losses, losses, global_steps, _ = sess.run(
            [self.each_vd_losses, self.losses, self.__global_step, train_op], feed_dict=feed_dict)
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
