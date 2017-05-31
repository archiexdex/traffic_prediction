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
tf.app.flags.DEFINE_string('log_dir', 'log/',
                           "summary directory")
tf.app.flags.DEFINE_integer('batch_size', 128,
                            "mini-batch size")
tf.app.flags.DEFINE_integer('total_epoches', 100,
                            "total training epoches")
tf.app.flags.DEFINE_integer('hidden_size', 28,
                            "size of LSTM hidden memory")
tf.app.flags.DEFINE_integer('rnn_layers', 2,
                            "number of stacked lstm")
tf.app.flags.DEFINE_integer('num_steps', 10,
                            "total steps of time")
tf.app.flags.DEFINE_boolean('is_float32', True,
                            "data type of the LSTM state, float32 if true, float16 otherwise")
tf.app.flags.DEFINE_float('learning_rate', 0.01,
                          "learning rate of RMSPropOptimizer")
tf.app.flags.DEFINE_float('decay_rate', 0.99,
                          "decay rate of RMSPropOptimizer")
tf.app.flags.DEFINE_float('momentum', 0.9,
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
        global_steps = tf.train.get_or_create_global_step(graph=graph)

        # read data
        train_raw_data = np.load(FLAGS.data_dir + "raw_data_6.npy")
        train_label_data = np.load(FLAGS.data_dir + "label_data_6.npy")
        test_raw_data = np.load(FLAGS.data_dir + "test_raw_data_6.npy")
        test_label_data = np.load(FLAGS.data_dir + "test_label_data_6.npy")

        # select flow from [density, flow, speed, weekday, time]
        train_raw_data = train_raw_data[:, :, :, 1]
        train_label_data = train_label_data[:, :, 1]
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
        loss_op = model.losses(logits=logits_op, labels=Y_ph)
        train_op = model.train(loss=loss_op, global_step=global_steps)

        # summary
        merged_op = tf.summary.merge_all()
        train_summary_writer = tf.summary.FileWriter(FLAGS.log_dir + 'train', graph=graph)
        test_summary_writer = tf.summary.FileWriter(FLAGS.log_dir + 'test', graph=graph)

        init = tf.global_variables_initializer()
        # saver
        saver = tf.train.Saver()

        # Session
        with tf.Session() as sess:
            sess.run(init)
            for epoch_steps in range(FLAGS.total_epoches):
                # shuffle
                raw_data_t = train_raw_data
                label_data_t = train_label_data
                concat = np.c_[raw_data_t.reshape(len(raw_data_t), -1),
                               label_data_t.reshape(len(label_data_t), -1)]
                np.random.shuffle(concat)
                train_raw_data = concat[:, :raw_data_t.size //
                                        len(raw_data_t)].reshape(raw_data_t.shape)
                train_label_data = concat[:, raw_data_t.size //
                                          len(raw_data_t):].reshape(label_data_t.shape)

                # training
                train_loss_sum = 0.0
                train_batches_amount = len(train_raw_data) // FLAGS.batch_size
                for i in range(train_batches_amount):
                    current_X_batch = train_raw_data[i:i + FLAGS.batch_size]
                    current_Y_batch = train_label_data[i:i + FLAGS.batch_size]
                    summary, _, loss_value, steps = \
                        sess.run([merged_op, train_op, loss_op, global_steps], feed_dict={
                                 X_ph: current_X_batch, Y_ph: current_Y_batch})
                    train_summary_writer.add_summary(
                        summary, global_step=steps)
                    train_loss_sum += loss_value

                # testing
                test_loss_sum = 0.0
                test_batches_amount = len(test_raw_data) // FLAGS.batch_size
                for i in range(test_batches_amount):
                    current_X_batch = test_raw_data[i:i + FLAGS.batch_size]
                    current_Y_batch = test_label_data[i:i + FLAGS.batch_size]
                    test_loss_value = sess.run(loss_op, feed_dict={
                        X_ph: current_X_batch, Y_ph: current_Y_batch})
                    test_loss_sum += test_loss_value

                # train mean ephoch loss
                train_mean_loss = train_loss_sum / train_batches_amount
                train_scalar_summary = tf.Summary()
                train_scalar_summary.value.add(
                    simple_value=train_mean_loss, tag="mean loss")
                train_summary_writer.add_summary(
                    train_scalar_summary, global_step=steps)
                train_summary_writer.flush()

                # test mean ephoch loss
                test_mean_loss = test_loss_sum / test_batches_amount
                test_scalar_summary = tf.Summary()
                test_scalar_summary.value.add(
                    simple_value=test_mean_loss, tag="mean loss")
                test_summary_writer.add_summary(
                    test_scalar_summary, global_step=steps)
                test_summary_writer.flush()

                print("ephoches: ", epoch_steps, "trainng loss: ", train_mean_loss,
                      "testing loss: ", test_mean_loss)

            # Save the variables to disk.
            save_path = saver.save(
                sess, FLAGS.checkpoints_dir, global_step=steps)
            print("Model saved in file: %s" % save_path)

        # TODO: https://www.tensorflow.org/api_docs/python/tf/trai/Supervisor
        # sv = Supervisor(logdir=FLAGS.checkpoints_dir)
        # with sv.managed_session(FLAGS.master) as sess:
        #     while not sv.should_stop():
        #         sess.run(<my_train_op>)


if __name__ == "__main__":
    if not os.path.exists(FLAGS.checkpoints_dir):
        os.makedirs(FLAGS.checkpoints_dir)
    tf.app.run()
