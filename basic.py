from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('log_dir', 'log',
                           "training log directory")
tf.app.flags.DEFINE_integer('batch_size', 128,
                            "mini-batch size")
tf.app.flags.DEFINE_integer('total_epoches', 10000,
                            "")
tf.app.flags.DEFINE_integer('hidden_size', 61,
                            "size of LSTM hidden memory")
tf.app.flags.DEFINE_integer('num_steps', 16,
                            "total steps of time")
tf.app.flags.DEFINE_boolean('is_float32', True,
                            "data type of the LSTM state, float32 if true, float16 otherwise")
tf.app.flags.DEFINE_float('learning_rate', 0.00025,
                          "learning rate of RMSPropOptimizer")
tf.app.flags.DEFINE_float('decay_rate', 0.99,
                          "decay rate of RMSPropOptimizer")
tf.app.flags.DEFINE_float('momentum', 0.0,
                          "momentum of RMSPropOptimizer")


class TFPModel(object):
    """
    The Traffic Flow Prediction Modle
    """

    def __init__(self, config, is_training=True):
        """
        Param:
            config:
        """
        self.is_training = is_training
        self.batch_size = config.batch_size
        self.hidden_size = config.hidden_size
        self.num_steps = config.num_steps
        if config.is_float32:
            self.data_type = tf.float32
        else:
            self.data_type = tf.float16
        self.learning_rate = config.learning_rate
        self.decay_rate = config.decay_rate
        self.momentum = config.momentum

    def lstm_cell(self):
        return tf.contrib.rnn.BasicLSTMCell(self.hidden_size, forget_bias=0.0, state_is_tuple=True,
                                            reuse=tf.get_variable_scope().reuse)

    def predict(self, inputs):
        """
        Param:
            inputs: [batch_size, time_step, milage] = [128, 15, 61]
        """
        logits_list = []
        cell = self.lstm_cell()

        state = cell.zero_state(FLAGS.batch_size, self.data_type)  # [128, 61]
        with tf.variable_scope('LSTM') as scope:
            for time_step in range(self.num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                cell_out, state = cell(
                    inputs[:, time_step, :], state, scope=scope)
                logits_list.append(cell_out)

        logits_list = tf.transpose(logits_list, perm=[1, 0, 2])
        return logits_list

    def loss_fn(self, logits, labels):
        """
        Param:
            logits:
            labels:
        """
        losses = tf.squared_difference(logits, labels)
        mean_loss_op = tf.reduce_mean(losses)
        return mean_loss_op

    def train(self, loss, global_step=None):
        """
        Param:
            loss:
        """
        train_op = tf.train.RMSPropOptimizer(
            self.learning_rate, self.decay_rate, self.momentum, 1e-10).minimize(loss, global_step=global_step)
        return train_op


class TestingConfig(object):
    """
    testing config
    """
    batch_size = FLAGS.batch_size
    hidden_size = FLAGS.hidden_size
    is_float32 = FLAGS.is_float32
    num_steps = FLAGS.num_steps
    learning_rate = FLAGS.learning_rate
    momentum = FLAGS.momentum
    decay_rate = FLAGS.decay_rate


def main(_):

    # TODO: read data into raw_data
    # raw_data as [time, milage, density]
    raw_data = []
    # TODO: maybe raw_data.shape[0] isn't the multiple of FLAGS.num_steps
    input_amount = raw_data.shape[0] / FLAGS.num_steps
    batches_in_epoch = input_amount / FLAGS.batch_size
    total_inputs = np.reshape(raw_data,
                              [input_amount,
                               FLAGS.num_steps,
                               raw_data.shape[1],
                               raw_data.shape[2]])
    total_inputs_X = total_inputs[:-128]
    total_inputs_Y = total_inputs[128:]
    total_inputs_concat = np.concatenate(
        (total_inputs_X, total_inputs_Y), axis=1)
    total_inputs_shuffle = np.random.shuffle(total_inputs_concat)
    total_inputs_split = np.hsplit( total_inputs_shuffle, 2)
    total_inputs_X = total_inputs_split[0]
    total_inputs_Y = total_inputs_split[1]

    model_config = TestingConfig()

    # TODO: data preprocessing X and T
    X = tf.placeholder(dtype=tf.float32, shape=[
                       None, FLAGS.num_steps, FLAGS.hidden_size], name='DFS')
    Y = tf.placeholder(dtype=tf.float32, shape=[
                       None, FLAGS.num_steps, FLAGS.hidden_size], name='DFS')

    model = TFPModel(model_config, is_training=True)
    logits = model.predict(X)
    losses = model.loss_fn(logits, Y)
    train_op = model.train(losses)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        for k in range(FLAGS.total_epoches):

            for i in range(batches_in_epoch):
                start_idx = i * FLAGS.batch_size
                end_idx = (i + 1) * FLAGS.batch_size
                current_X_batch = total_inputs_X[start_idx:end_idx]
                current_Y_batch = total_inputs_Y[start_idx:end_idx]

                _, loss_value = sess.run([train_op, losses], feed_dict={X: current_X_batch, Y: current_Y_batch})

                # TODO: https://www.tensorflow.org/api_docs/python/tf/trai/Supervisor
                # sv = Supervisor(logdir=FLAGS.log_dir)
                # with sv.managed_session(FLAGS.master) as sess:
                #     while not sv.should_stop():
                #         sess.run(<my_train_op>)


if __name__ == "__main__":
    tf.app.run()
