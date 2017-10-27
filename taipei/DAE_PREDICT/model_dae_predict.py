
# TODO input normalization or batch normalization
# TODO add MAPE evaluate fn
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class DAE_TFP_Model(object):
    """
    A : DAE = denoising autoencoder
    B : TFP = traffic flow prediction
    The model is use to predict traffic flow of the future.
    we could experiment different combinations of follows
        'trainA_trainB' : fine tune A and B together
        'fixA_trainB' : fix A to train B only
        'noA_trainB' : no A model train/predict B model only
    """

    def __init__(self, config, graph=None):
        """
        Params
        ------ 
            config : 
                * data_dir : data directory
                * checkpoints_dir : training checkpoints directory
                * log_dir : summary directory
                * batch_size : mini-batch size
                * total_epoches : total training epoches
                * save_freq : number of epoches to saving model
                * total_interval : total steps of time
                * learning_rate : learning rate of AdamOptimizer
                * input_shape : input data shape
                * label_shape : label data shape
                * train_mode : training mode
                * if_dae_recover_all : True, dae output as predict input. False, dae output on mask position + original data.
            graph : default graph
        """
        if config.train_mode != 'noA_trainB':
            # trainA_trainB or fixA_trainB
            # build up the DAE graph
            tf.train.import_meta_graph(
                config.restore_dae_path + '.meta')
            # dae_output: A model's last tensor (recovered data) as the input of B model
            # dae_output = graph.get_tensor_by_name('DAE/deconv2/sub:0')
            dae_output = graph.get_tensor_by_name('recover_logits_scale/add:0')

        self.__global_step = tf.train.get_or_create_global_step(graph=graph)

        with tf.variable_scope('PREDICT'):
            self.__batch_size = config.batch_size
            self.__log_dir = config.log_dir
            self.__learning_rate = config.learning_rate
            self.__label_shape = config.label_shape
            # model IO : self.__X_ph -> A model -> B model -> self.__Y_ph
            if config.train_mode != 'noA_trainB':
                # trainA_trainB or fixA_trainB
                self.__X_ph = graph.get_tensor_by_name('corrupt_data:0')
                if not config.if_dae_recover_all:
                    dae_output = self.dae_recover_mask_only(
                        dae_output, self.__X_ph)
                else:
                    dae_output = tf.concat(
                        [self.__X_ph[:, :, :, 0:1], dae_output, self.__X_ph[:, :, :, 4:6]], axis=-1)
                tfp_input = dae_output
            else:
                self.__X_ph = tf.placeholder(dtype=tf.float32, shape=[
                    None, config.input_shape[1], config.input_shape[2], config.input_shape[3]], name='raw_data')
                tfp_input = self.__X_ph
            self.__Y_ph = tf.placeholder(dtype=tf.float32, shape=[
                None, config.label_shape[1], config.label_shape[2]], name='label_data')
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.__learning_rate)
            with tf.variable_scope('is_training'):
                self.__logits_train = self.__inference(tfp_input, config.dropout, is_training=True)
                tf.get_variable_scope().reuse_variables()
                self.__logits_test  = self.__inference(tfp_input, config.dropout, is_training=False)
            self.__each_vd_losses_train, self.__losses_train = self.__loss_function(
                self.__logits_train, self.__Y_ph)
            self.__each_vd_losses_test, self.__losses_test = self.__loss_function(
                self.__logits_test, self.__Y_ph)
            self.__train_loss_summary = tf.summary.scalar(
                'loss', self.__losses_train)
            # fix A train B only
            self.__train_op = optimizer.minimize(
                self.__losses_train, var_list=self.__get_var_list(), global_step=self.__global_step)
            # train A+B
            self.__train_all_op = optimizer.minimize(
                self.__losses_train, global_step=self.__global_step)

            # summary writer
            self.__summary_writer = tf.summary.FileWriter(
                self.__log_dir + 'PREDICT_train', graph=graph)

    def dae_recover_mask_only(self, dae_out, origin_data):
        """ data recovering : origin_data add dae_out when missing
        Params
        ------
            dae_out : float, shape=[b, vds, intervals, features]
                output of DAE, imputed data, recovered data
            origin_data : float, shape=[b, vds, intervals, features]
                input of DAE, raw data, corrupted data (contain missing value)
        note
        ----
            features : shape=[6]
                [time, density, flow, speed, weekday, missing_mask]
        """
        missing_mask = tf.cast(origin_data[:, :, :, -1], tf.bool)
        stacked_missing_mask = tf.stack(
            [missing_mask for _ in range(3)], axis=-1)
        # data imputation from DAE output on those missing value (density, flow, speed)
        imputed_data = tf.where(stacked_missing_mask,
                                dae_out, origin_data[:, :, :, 1:4])
        result = tf.concat(
            [origin_data[:, :, :, 0:1], imputed_data, origin_data[:, :, :, 4:6]], axis=-1)
        return result

    def __get_var_list(self):
        """ To get the TFP model's trainable variables.
        """
        trainable_V = tf.trainable_variables()
        theta_PREDICT = []
        for _, v in enumerate(trainable_V):
            if v.name.startswith('PREDICT'):
                theta_PREDICT.append(v)
        return theta_PREDICT

    def __leakyrelu(self, features, alpha=0.2):
        """
        """
        return tf.maximum(features, features * alpha)
    
    def __swish(self, features):
        return tf.nn.sigmoid(features) * features

    def __inference(self, inputs, dropout, is_training):
        """
        Params
        ------
            inputs : float32, placeholder, shape=[b, vds, intervals, features]
                output of DAE, imputed data, recovered data
        Return
        ------
            fully2 : float, shape=[None, target_vds, intervals]
        note
        ----
            features : shape=[6]
                [time, density, flow, speed, weekday, missing_mask]
        """
        print("inputs:", inputs)
        with tf.variable_scope('conv1') as scope:
            kernel_init = tf.truncated_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            bias_init = tf.random_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            conv1 = tf.layers.conv2d(inputs=inputs, filters=64, kernel_size=[3, 3],
                                     strides=1, padding='same', activation=self.__leakyrelu,
                                     kernel_initializer=kernel_init, bias_initializer=bias_init, reuse=scope.reuse)
            print("conv1:", conv1)
        # with tf.variable_scope('conv1_2') as scope:
        #     kernel_init = tf.truncated_normal_initializer(
        #         mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
        #     bias_init = tf.random_normal_initializer(
        #         mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
        #     conv1_2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=[3, 3],
        #                                strides=1, padding='same', activation=self.__leakyrelu,
        #                                kernel_initializer=kernel_init, bias_initializer=bias_init, reuse=scope.reuse)
        #     print("conv1_2:", conv1_2)
        with tf.variable_scope('conv2') as scope:
            kernel_init = tf.truncated_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            bias_init = tf.random_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            conv2 = tf.layers.conv2d(inputs=conv1, filters=128, kernel_size=[3, 3],
                                     strides=2, padding='same', activation=self.__leakyrelu,
                                     kernel_initializer=kernel_init, bias_initializer=bias_init, reuse=scope.reuse)
            print("conv2:", conv2)
        # with tf.variable_scope('conv3') as scope:
        #     kernel_init = tf.truncated_normal_initializer(
        #         mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
        #     bias_init = tf.random_normal_initializer(
        #         mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
        #     conv3 = tf.layers.conv2d(inputs=conv2, filters=128, kernel_size=[3, 3],
        #                              strides=1, padding='same', activation=self.__leakyrelu,
        #                              kernel_initializer=kernel_init, bias_initializer=bias_init, reuse=scope.reuse)
        #     print("conv3:", conv3)
        with tf.variable_scope('conv3') as scope:
            kernel_init = tf.truncated_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            bias_init = tf.random_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            conv3 = tf.layers.conv2d(inputs=conv2, filters=256, kernel_size=[3, 3],
                                     strides=2, padding='same', activation=self.__leakyrelu,
                                     kernel_initializer=kernel_init, bias_initializer=bias_init, reuse=scope.reuse)
            print("conv3:", conv3)
        # with tf.variable_scope('conv4_2') as scope:
        #     kernel_init = tf.truncated_normal_initializer(
        #         mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
        #     bias_init = tf.random_normal_initializer(
        #         mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
        #     conv4_2 = tf.layers.conv2d(inputs=conv4, filters=256, kernel_size=[3, 3],
        #                                strides=1, padding='same', activation=self.__leakyrelu,
        #                                kernel_initializer=kernel_init, bias_initializer=bias_init, reuse=scope.reuse)
        #     print("conv4_2:", conv4_2)
        with tf.variable_scope('fully1') as scope:
            kernel_init = tf.truncated_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            bias_init = tf.random_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            flat = tf.contrib.layers.flatten(inputs=conv3)
            fully1 = tf.contrib.layers.fully_connected(flat,
                                                       1024,
                                                       activation_fn=self.__leakyrelu,
                                                       weights_initializer=kernel_init,
                                                       biases_initializer=bias_init, reuse=scope.reuse,scope=scope)
            print("fully1:", fully1)
        
        with tf.variable_scope('dropout') as scope:
            fully1 = tf.layers.dropout(fully1, rate=dropout, training=is_training)
            print("fully1_drop:", fully1)
        
        with tf.variable_scope('fully2') as scope:
            kernel_init = tf.truncated_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            bias_init = tf.random_normal_initializer(
                mean=0.0, stddev=0.01, seed=None, dtype=tf.float32)
            fully2 = tf.contrib.layers.fully_connected(fully1,
                                                       self.__label_shape[1] *
                                                       self.__label_shape[2],
                                                       activation_fn=self.__leakyrelu,
                                                       weights_initializer=kernel_init,
                                                       biases_initializer=bias_init, reuse=scope.reuse,scope=scope)
            print("fully2:", fully2)

        with tf.variable_scope('reshape') as scope:
            reshaped = tf.reshape(
                fully2, [-1, 56, 4])
            print("reshape:", reshaped)

        return reshaped

    def __loss_function(self, logits, labels):
        """
        Params
        ------
            logits : float, shape=[None, num_target_vds]
                self.__inference's output
            labels : float, placeholder, shape=[None, target_vds, intevals]
        Return
        ------
            vd_losses : float, shape=[num_vds]
                every vd's loss in one step
            l2_mean_loss : float, shape=[]
                mean loss for backprop
        """
        with tf.name_scope('l2_loss'):
            vd_losses = tf.squared_difference(logits, labels)
            vd_mean_losses = tf.reduce_mean(vd_losses, axis=0)
            l2_mean_loss = tf.reduce_mean(vd_losses)
        print(l2_mean_loss)
        return vd_mean_losses, l2_mean_loss

    def step(self, sess, inputs, labels, train_mode):
        """
        Params
        ------
            sess : tf.Session()
            inputs : float, shape=[b, vds, intervals, features]
                input of DAE, raw data, corrupted data (contain missing value)
            labels : float, shape=[None, target_vds, intevals]
                label data
            train_mode : bool
                'trainA_trainB' -> update A+B. 'fixA_trainB' -> update B only. 'noA_trainB' -> update B. (no A)
        Return
        ------
            each_vd_losses: float, shape=[num_vds]
                every vd's loss in one step
            losses: float, shape=[]
                mean loss for backprop
            global_steps: int, shape=[]
                training steps
        """
        feed_dict = {self.__X_ph: inputs,
                     self.__Y_ph: labels}
        if train_mode == 'fixA_trainB':
            train_op = self.__train_op
        else:
            # train_mode : 'trainA_trainB' or 'noA_trainB'
            train_op = self.__train_all_op
        summary, each_vd_losses, losses, global_steps, _ = sess.run(
            [self.__train_loss_summary, self.__each_vd_losses_train, self.__losses_train, self.__global_step, train_op], feed_dict=feed_dict)
        self.__summary_writer.add_summary(
            summary, global_step=global_steps)
        return each_vd_losses, losses, global_steps

    def compute_loss(self, sess, inputs, labels):
        """
        Return
        ------
            each_vd_losses : float, shape=[num_vds]
                every vd's loss in one step
            losses : float, shape=[]
                mean loss for backprop
        """
        feed_dict = {self.__X_ph: inputs,
                     self.__Y_ph: labels}
        each_vd_losses, losses = sess.run(
            [self.__each_vd_losses_test, self.__losses_test], feed_dict=feed_dict)
        return each_vd_losses, losses

    def predict(self, sess, inputs):
        """
        Return
        ------
            prediction : float, shape=[None, target_vds, intervals]
        """
        feed_dict = {self.__X_ph: inputs}
        prediction = sess.run(self.__logits_test, feed_dict=feed_dict)
        return prediction


if __name__ == '__main__':
    pass
