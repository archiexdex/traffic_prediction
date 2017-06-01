from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import model_lstm

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', 'data/',
                           "data directory")
tf.app.flags.DEFINE_string('checkpoints_dir', 'checkpoints/',
                           "training checkpoints directory")
tf.app.flags.DEFINE_string('log_dir', 'test_log/',
                           "summary directory")
tf.app.flags.DEFINE_integer('batch_size', 1,
                            "mini-batch size")
tf.app.flags.DEFINE_integer('total_epoches', 0,
                            "total training epoches")
tf.app.flags.DEFINE_integer('hidden_size', 28,
                            "size of LSTM hidden memory")
tf.app.flags.DEFINE_integer('rnn_layers', 1,
                            "number of stacked lstm")
tf.app.flags.DEFINE_integer('num_steps', 10,
                            "total steps of time")
tf.app.flags.DEFINE_boolean('is_float32', True,
                            "data type of the LSTM state, float32 if true, float16 otherwise")
tf.app.flags.DEFINE_float('learning_rate', 0,
                          "learning rate of RMSPropOptimizer")
tf.app.flags.DEFINE_float('decay_rate', 0,
                          "decay rate of RMSPropOptimizer")
tf.app.flags.DEFINE_float('momentum', 0,
                          "momentum of RMSPropOptimizer")


def read_file(filename, vec, week_list, time, week, st, ed):
    filename = "../../VD_data/mile_base/" + filename
    with open(filename, "rb") as binaryfile:
        binaryfile.seek(0)
        ptr = binaryfile.read(4)

        data_per_day = 1440
        VD_size = int.from_bytes(ptr, byteorder='little')
        ptr = binaryfile.read(4)
        day_max = int.from_bytes(ptr, byteorder='little')

        # initialize list
        dis = int((ed - st) * 2 + 1)
        t = len(vec)
        for i in range(day_max):
            vec.append([0] * dis)
            week_list.append([0] * dis)
            time.append([0] * dis)

        index = 0
        for i in range(VD_size):

            if st <= i / 2 and i / 2 <= ed:
                for j in range(day_max):
                    ptr = binaryfile.read(2)
                    tmp = int.from_bytes(ptr, byteorder='little')
                    vec[t + j][index] = tmp
                    week_list[t + j][index] = (week +
                                               int(j / data_per_day)) % 7
                    time[t + j][index] = j % data_per_day
                index = index + 1
            elif ed < i / 2:
                break
            else:
                binaryfile.read(2)


class TestingConfig(object):
    """
    testing config
    """

    def __init__(self):
        self.data_dir = FLAGS.data_dir
        self.checkpoints_dir = FLAGS.checkpoints_dir
        self.log_dir = FLAGS.log_dir
        self.batch_size = FLAGS.batch_size
        self.total_epoches = FLAGS.total_epoches
        self.hidden_size = FLAGS.hidden_size
        self.rnn_layers = FLAGS.rnn_layers
        self.num_steps = FLAGS.num_steps
        self.is_float32 = FLAGS.is_float32
        self.learning_rate = FLAGS.learning_rate
        self.decay_rate = FLAGS.decay_rate
        self.momentum = FLAGS.momentum

    def show(self):
        print("data_dir:", self.data_dir)
        print("checkpoints_dir:", self.checkpoints_dir)
        print("log_dir:", self.log_dir)
        print("batch_size:", self.batch_size)
        print("total_epoches:", self.total_epoches)
        print("hidden_size:", self.hidden_size)
        print("rnn_layers:", self.rnn_layers)
        print("num_steps:", self.num_steps)
        print("is_float32:", self.is_float32)
        print("learning_rate:", self.learning_rate)
        print("decay_rate:", self.decay_rate)
        print("momentum:", self.momentum)


def main(_):
    with tf.get_default_graph().as_default() as graph:

        # read data [amount, num_steps, mileage, dfswt] == [None, 10, 28, 5]
        test_raw_data = np.load(FLAGS.data_dir + "test_raw_data_6.npy")
        test_label_data = np.load(FLAGS.data_dir + "test_label_data_6.npy")

        # select flow from [density, flow, speed, weekday, time]
        test_raw_data = test_raw_data[:, :, :, 1]
        test_label_data = test_label_data[:, :, 1]

        # placeholder
        X_ph = tf.placeholder(dtype=tf.float32, shape=[
                              FLAGS.batch_size, FLAGS.num_steps, 28], name='input_data')
        Y_ph = tf.placeholder(dtype=tf.float32, shape=[
                              FLAGS.batch_size, 28], name='label_data')

        # config setting
        config = TestingConfig()
        config.show()

        # model
        model = model_lstm.TFPModel(config, is_training=True)
        logits_op = model.inference(inputs=X_ph)
        losses_op = model.losses(logits=X_ph, labels=Y_ph)

        # summary
        labels_summary_writer = tf.summary.FileWriter(
            FLAGS.log_dir + 'observation', graph=graph)
        logits_summary_writer = tf.summary.FileWriter(
            FLAGS.log_dir + 'prediction', graph=graph)

        init = tf.global_variables_initializer()
        # saver
        saver = tf.train.Saver()

        # Session
        with tf.Session() as sess:
            sess.run(init)

            saver.restore(sess, FLAGS.checkpoints_dir + '-985000')
            print("Successully restored!!")

            # testing
            test_loss_sum = 0.0
            # for i, _ in enumerate(test_raw_data):
            for i in range(60*24):
                offset = i + 60*24*4
                current_X_batch = test_raw_data[offset:offset + 1]
                current_Y_batch = test_label_data[offset:offset + 1]
                predicted_value, losses_value = sess.run([logits_op, losses_op], feed_dict={
                    X_ph: current_X_batch, Y_ph: current_Y_batch})
                test_loss_sum += losses_value

                labels_scalar_summary = tf.Summary()
                labels_scalar_summary.value.add(
                    simple_value=current_Y_batch[0][15], tag="cmp")
                labels_summary_writer.add_summary(
                    labels_scalar_summary, global_step=i)
                labels_summary_writer.flush()

                logits_scalar_summary = tf.Summary()
                logits_scalar_summary.value.add(
                    simple_value=predicted_value[0][15], tag="cmp")
                logits_summary_writer.add_summary(
                    logits_scalar_summary, global_step=i)
                logits_summary_writer.flush()

            # test mean loss
            train_mean_loss = test_loss_sum / 1440

            print("testing mean loss: ", train_mean_loss)

        # TODO: https://www.tensorflow.org/api_docs/python/tf/trai/Supervisor
        # sv = Supervisor(logdir=FLAGS.checkpoints_dir)
        # with sv.managed_session(FLAGS.master) as sess:
        #     while not sv.should_stop():
        #         sess.run(<my_train_op>)


if __name__ == "__main__":
    if not os.path.exists(FLAGS.checkpoints_dir):
        os.makedirs(FLAGS.checkpoints_dir)
    tf.app.run()
