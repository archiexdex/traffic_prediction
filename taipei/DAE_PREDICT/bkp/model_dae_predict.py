from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import utils
import model_dae
import model_predict


class DAE_PREDICT_MODEL(object):
    """
    A: DAE model
    B: PREDICT model
    Wrapper for DAE+PREDICT model

    USAGE:
    ------
    * step_A: train DAE model only
    * step_B: fixed DAE, train PREDICT model only
    * step_AB: train DAE+PREDICT model together
    """

    def __init__(self, A_config, B_config, graph):
        self.__global_step = tf.train.get_or_create_global_step(graph=graph)

        self.__corrupt_data = tf.placeholder(dtype=tf.float32, shape=[
            A_config.batch_size, A_config.input_shape[1], A_config.input_shape[2], A_config.input_shape[3]], name='corrupt_data')
        self.__raw_data = tf.placeholder(dtype=tf.float32, shape=[
            A_config.batch_size, A_config.input_shape[1], A_config.input_shape[2], 3], name='raw_data')
        self.__label_data = tf.placeholder(dtype=tf.float32, shape=[
            None, B_config.test_shape[1], B_config.test_shape[2]], name='label_data')

        self.__A_model = model_dae.DAEModel(
            A_config, self.__corrupt_data, self.__raw_data, graph=graph)
        self.__B_model = model_predict.TFPModel(
            B_config, self.__corrupt_data, self.__label_data, self.__A_model._logits, graph=graph)

    def step_A(self, sess, inputs, labels):
        return self.__A_model.step(sess, inputs, labels)

    def step_B(self, sess, inputs, labels):
        return self.__B_model.step(sess, inputs, labels, if_train_all=False)

    def step_AB(self, sess, inputs, labels):
        return self.__B_model.step(sess, inputs, labels, if_train_all=True)


class A_TestingConfig(object):
    """
    A testing config
    """

    def __init__(self):
        self.filter_numbers = [32, 64, 128]
        self.filter_strides = [1, 2, 2]
        self.batch_size = 256
        self.total_epoches = 10
        self.learning_rate = 0.001
        self.log_dir = "test_log_dir/"
        self.input_shape = [30208, 45, 12, 6]
        self.if_label_normed = True
        self.if_mask_only = True


class B_TestingConfig(object):
    """
    B testing config
    """

    def __init__(self):
        self.log_dir = "test_log_dir/"
        self.batch_size = 256
        self.total_epoches = 10
        self.learning_rate = 0.001
        self.num_gpus = 1
        self.train_shape = [30208, 45, 12, 6]
        self.test_shape = [30208, 18, 4]


def test():
    with tf.Graph().as_default() as g:
        A_config = A_TestingConfig()
        B_config = B_TestingConfig()
        model = DAE_PREDICT_MODEL(A_config, B_config, graph=g)
        # train
        X = np.zeros(shape=[256, 45, 12, 6])
        Y = np.zeros(shape=[256, 18, 4])
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(100):
                _, _, global_steps = model.step_A(sess, X, X[:, :, :, 1:4])
                print('global_steps', global_steps)
                _, _, global_steps = model.step_B(sess, X, Y)
                print('global_steps', global_steps)
                _, _, global_steps = model.step_AB(sess, X, Y)
                print('global_steps', global_steps)


if __name__ == '__main__':
    test()
