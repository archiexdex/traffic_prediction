from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import utils


class DAEModel(object):
    """ Model of Denoising Convolutional AutoEncoder for data imputation
    # TODO list
        * weights/ bias initilizer
        * filter amount
        * multi-gpu
        * self.is_training 
        * predict and visulize
    """

    def __init__(self, config, corrupt_data, raw_data, graph):
        """ build up the whole tensorflow computational graph
        Params
        ------
        config : class, hyper-perameters
            * filter_numbers : int, list, the number of filters for each conv and decov in reversed order
            * filter_strides : int, list, the value of stride for each conv and decov in reversed order
            * batch_size : int, mini batch size
            * log_dir : string, the path to save training summary
            * learning_rate : float, adam's learning rate
            * input_shape : int, list, e.g. [?, nums_VD, nums_Interval, nums_features]
        graph : tensorflow default graph
            for summary writer and __global_step init
        """
        # use for loss recovering
        self.Norm_er = utils.Norm()
        # hyper-parameters
        self.filter_numbers = config.filter_numbers
        self.filter_strides = config.filter_strides
        self.batch_size = config.batch_size
        self.log_dir = config.log_dir
        self.learning_rate = config.learning_rate
        self.input_shape = config.input_shape
        self.if_label_normed = config.if_label_normed
        self.if_mask_only = config.if_mask_only

        # steps
        self.__global_step = tf.train.get_or_create_global_step(graph=graph)
        # data
        self.__corrupt_data = corrupt_data
        self.__raw_data = raw_data
        # get missing mask from input
        self.__missing_mask = self.__corrupt_data[:, :, :, -1]
        # model
        self._logits = self.__inference(
            self.__corrupt_data, self.filter_numbers, self.filter_strides)
        print(self._logits)
        exit()
        self.__loss, self.__sep_loss = self.__loss_function(
            self._logits, self.__raw_data)
        # add to summary
        tf.summary.scalar('loss', self.__loss)
        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate)
        # print(tf.trainable_variables())
        self.__train_op = optimizer.minimize(
            self.__loss, var_list=self.__get_var_list(), global_step=self.__global_step)

        # summary
        self.__merged_op = tf.summary.merge_all()
        # summary writer
        self.train_summary_writer = tf.summary.FileWriter(
            self.log_dir + 'DAE_train', graph=graph)

    def __get_var_list(self):
        """ 
        """
        trainable_V = tf.trainable_variables()
        theta_DAE = []
        for _, v in enumerate(trainable_V):
            if v.name.startswith('DAE'):
                theta_DAE.append(v)
        return theta_DAE

    def __inference(self, corrupt_data, filter_numbers, filter_strides):
        """ construct the AutoEncoder model
        Params
        ------
        corrupt_data : placeholder, shape=[batch_size, nums_vd, nums_interval, features]
            get by randomly corrupting raw data
        filter_numbers : int, list
            the number of filters for each conv and decov in reversed order. e.g. [32, 64, 128]
        filter_strides : int, list, 
            the value of stride for each conv and decov in reversed order. e.g. [1, 2, 2]

        Return
        ------
        output : the result of AutoEndoer, shape is same as 'corrupt_data'
        """
        def lrelu(x, alpha=0.3):
            return tf.nn.relu(x) - alpha * tf.nn.relu(-x)
        print("corrupt_data:", corrupt_data)
        shapes_list = []
        with tf.variable_scope("DAE") as out_scope:
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
                    if layer_id == len(shapes_list) - 1:
                        # only regress 3 dims as [d, f, s]
                        out_filter_amount = 3
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

        if self.if_label_normed:
            with tf.name_scope('recover_logits_scale'):
                output = self.Norm_er.logits_recover(output)
        return output

    def __loss_function(self, logits, labels):
        """ l2 function (mean square error)
        Params
        ------
        logits : tensor, from __inference()
        labels : placeholder, raw data

        Return
        ------
        l2_mean_loss : tensor 
            MSE of one batch
        sep_mean_loss : tensor, shape = [batch_size, num_vds, num_intervals, 3]
            the mean loss of each dimension (density, flow, speed)

        """
        if self.if_mask_only:
            stacked__missing_mask = tf.stack(
                [self.__missing_mask for _ in range(3)], axis=-1)
            logits = tf.multiply(logits, stacked__missing_mask)
            labels = tf.multiply(labels, stacked__missing_mask)
            num_missing = tf.reduce_sum(self.__missing_mask)
            with tf.name_scope('l2_loss'):
                vd_losses = tf.squared_difference(logits, labels)
                sep_mean_loss = tf.reduce_sum(tf.reshape(vd_losses, shape=[
                    self.batch_size * self.input_shape[1] * self.input_shape[2], 3]), axis=0) / num_missing
                l2_mean_loss = tf.reduce_mean(sep_mean_loss)
        else:
            with tf.name_scope('l2_loss'):
                vd_losses = tf.squared_difference(logits, labels)
                sep_mean_loss = tf.reduce_mean(tf.reshape(vd_losses, shape=[
                    self.batch_size * self.input_shape[1] * self.input_shape[2], 3]), axis=0)
                l2_mean_loss = tf.reduce_mean(sep_mean_loss)

        print('l2_mean_loss:', l2_mean_loss)
        print('sep_mean_loss:', sep_mean_loss)

        return l2_mean_loss, sep_mean_loss

    def step(self, sess, inputs, labels):
        """ train one batch and update one time
        Params
        ------
        sess : tf.Session()
        inputs: corrupted data, shape=[batch_size, nums_vd, nums_interval, features]
        labels: raw data, shape=[batch_size, nums_vd, nums_interval, features]

        Return
        ------
        loss : float 
            MSE of one batch
        sep_loss : float, shape=[3]
            MSE of each dimensions
        global_steps : 
            the number of batches have been trained
        """
        feed_dict = {self.__corrupt_data: inputs, self.__raw_data: labels}
        summary, loss, sep_loss, global_steps, _ = sess.run(
            [self.__merged_op, self.__loss, self.__sep_loss, self.__global_step, self.__train_op], feed_dict=feed_dict)
        # write summary
        self.train_summary_writer.add_summary(
            summary, global_step=global_steps)
        return loss, sep_loss, global_steps

    def compute_loss(self, sess, inputs, labels):
        """ compute loss
        Params
        ------
        sess : tf.Session()
        inputs: corrupted data, shape=[batch_size, nums_vd, nums_interval, features]
        labels: raw data, shape=[batch_size, nums_vd, nums_interval, features]

        Return
        ------
        loss : float 
            MSE of one batch
        sep_loss : float, shape=[3]
            MSE of each dimensions
        """
        feed_dict = {self.__corrupt_data: inputs, self.__raw_data: labels}
        # loss = sess.run(self.__loss, feed_dict=feed_dict)
        loss, sep_loss = sess.run(
            [self.__loss, self.__sep_loss], feed_dict=feed_dict)
        return loss, sep_loss

    def predict(self, sess, inputs):
        """ recover the inputs (corrupted data)
        Params
        ------
        sess : tf.Session()
        inputs: corrupted data, shape=[batch_size, nums_vd, nums_interval, features]

        Return
        ------
        result : raw data, shape=[batch_size, nums_vd, nums_interval, features]
        """
        feed_dict = {self.__corrupt_data: inputs}
        result = sess.run(self._logits, feed_dict=feed_dict)
        return result
