from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class DCAEModel(object):
    """
    Model of Denoising Convolutional AutoEncoder for data imputation
    # TODO list
        * weights/ bias initilizer
        * filter amount
        * multi-gpu
        * self.is_training 
        * predict and visulize
    """

    def __init__(self, config, graph):
        """build up the whole tf graph
        TODO
        Params
        ------
        config : class, hyper-perameters
            * filter_numbers : list
            * filter_strides : list
            * batch_size : 
            * log_dir :
            * learning_rate : 
            * input_shape : 
            * label_shape : 

        graph : tensorflow default graph
        """
        # hyper-parameters
        self.filter_numbers = config.filter_numbers
        self.filter_strides = config.filter_strides
        self.batch_size = config.batch_size
        self.log_dir = config.log_dir
        self.learning_rate = config.learning_rate
        self.input_shape = config.input_shape
        # steps
        self.global_step = tf.train.get_or_create_global_step(graph=graph)
        # data
        self.corrupt_data = tf.placeholder(dtype=tf.float32, shape=[
            self.batch_size, self.input_shape[1], self.input_shape[2], self.input_shape[3]], name='corrupt_data')
        self.raw_data = tf.placeholder(dtype=tf.float32, shape=[
            self.batch_size, self.input_shape[1], self.input_shape[2], 3], name='raw_data')
        # model
        self.logits = self.inference(
            self.corrupt_data, self.filter_numbers, self.filter_strides)
        self.loss = self.loss_function(
            self.logits, self.raw_data)
        # add to summary
        tf.summary.scalar('loss', self.loss)
        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate)
        # print(tf.trainable_variables())
        self.train_op = optimizer.minimize(
            self.loss, global_step=self.global_step)

        # summary
        self.merged_op = tf.summary.merge_all()
        # summary writer
        self.train_summary_writer = tf.summary.FileWriter(
            self.log_dir + 'train', graph=graph)

    def inference(self, corrupt_data, filter_numbers, filter_strides):
        """
        TODO
        filter_numbers : [32, 64, 128]
        filter_strides : [1, 2, 2]
        """
        def lrelu(x, alpha=0.3):
            return tf.nn.relu(x) - alpha * tf.nn.relu(-x)
        print("corrupt_data:", corrupt_data)
        shapes_list = []
        # encoder
        current_input = corrupt_data
        for layer_id, out_filter_amount in enumerate(filter_numbers):
            with tf.variable_scope('conv' + str(layer_id)) as scope:
                # shape
                shapes_list.append(current_input.get_shape().as_list())
                in_filter_amount = current_input.get_shape().as_list()[3]
                # init
                kernel_init = tf.truncated_normal_initializer(
                    mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
                bias_init = tf.random_normal_initializer(
                    mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
                W = tf.get_variable(name='weights', shape=[
                                    3, 3, in_filter_amount, out_filter_amount], initializer=kernel_init)
                b = tf.get_variable(
                    name='bias', shape=out_filter_amount, initializer=bias_init)
                # conv
                stide = filter_strides[layer_id]
                output = lrelu(
                    tf.add(tf.nn.conv2d(
                        input=current_input, filter=W, strides=[1, stide, stide, 1], padding='SAME'), b))
                current_input = output
                print(scope.name, output)

        # print('shapes_list:', shapes_list)
        # reverse order for decoder part
        shapes_list.reverse()
        filter_strides.reverse()

        # decoder
        for layer_id, layer_shape in enumerate(shapes_list):
            with tf.variable_scope('deconv' + str(layer_id)) as scope:
                # shape
                in_filter_amount = current_input.get_shape().as_list()[3]
                if layer_id == len(shapes_list)-1:
                    out_filter_amount = 3 # only regress 3 dims as [d, f, s] 
                else:
                    out_filter_amount = layer_shape[3]               
                # init
                kernel_init = tf.truncated_normal_initializer(
                    mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
                bias_init = tf.random_normal_initializer(
                    mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
                W = tf.get_variable(name='weights', shape=[
                                    3, 3, out_filter_amount, in_filter_amount], initializer=kernel_init)
                b = tf.get_variable(
                    name='bias', shape=out_filter_amount, initializer=bias_init)
                # deconv
                stide = filter_strides[layer_id]
                output = lrelu(
                    tf.add(tf.nn.conv2d_transpose(
                        value=current_input, filter=W, output_shape=tf.stack([layer_shape[0], layer_shape[1], layer_shape[2], out_filter_amount]), strides=[1, stide, stide, 1], padding='SAME'), b))
                current_input = output
                print(scope.name, output)

        return output

    def loss_function(self, logits, labels):
        """
        TODO
        """
        with tf.name_scope('l2_loss'):
            vd_losses = tf.squared_difference(logits, labels)
            l2_mean_loss = tf.reduce_mean(vd_losses)
        print('l2_mean_loss:', l2_mean_loss)
        return l2_mean_loss

    def step(self, sess, inputs, labels):
        """
        TODO
        """
        feed_dict = {self.corrupt_data: inputs, self.raw_data: labels}
        summary, loss, global_steps, _ = sess.run(
            [self.merged_op, self.loss, self.global_step, self.train_op], feed_dict=feed_dict)
        # write summary
        self.train_summary_writer.add_summary(
            summary, global_step=global_steps)
        return loss, global_steps

    def compute_loss(self, sess, inputs, labels):
        """
        TODO
        """
        feed_dict = {self.corrupt_data: inputs, self.raw_data: labels}
        loss = sess.run(self.loss, feed_dict=feed_dict)
        return loss


class TestingConfig(object):
    """
    testing config
    """

    def __init__(self):
        self.filter_numbers = [32, 64, 128]
        self.filter_strides = [1, 2, 2]
        self.batch_size = 256
        self.total_epoches = 10
        self.learning_rate = 0.001
        self.log_dir = "FLAGS.log_dir/"
        self.input_shape = [256, 100, 12, 5]


def test():
    with tf.Graph().as_default() as g:
        config = TestingConfig()
        model = DCAEModel(config, graph=g)
        # train
        X = np.zeros(shape=[256, 100, 12, 5])
        Y = np.zeros(shape=[256, 100, 12, 5])
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(config.total_epoches):
                loss, global_steps = model.step(sess, X, Y)
                print('loss', loss)


if __name__ == '__main__':
    test()
