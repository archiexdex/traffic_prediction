from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import model_lstm

raw_data_name = "test_batch_no_over_data_mile_15_28.5_total_60_predict_1_5.npy"
label_data_name = "test_label_no_over_data_mile_15_28.5_total_60_predict_1_5.npy"

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', '/home/nctucgv/Documents/TrafficVis_Run/src/traffic_flow_detection/',
                           "data directory")
tf.app.flags.DEFINE_string('checkpoints_dir', 'backlog_new/' + raw_data_name[11:-4] + '/checkpoints/',
                           "training checkpoints directory")
tf.app.flags.DEFINE_string('log_dir', 'backlog_new/' + raw_data_name[11:-4] + '/test_log/',
                           "summary directory")
tf.app.flags.DEFINE_integer('batch_size', 1,
                            "mini-batch size")
tf.app.flags.DEFINE_integer('total_epoches', 0,
                            "total training epoches")
tf.app.flags.DEFINE_integer('hidden_size', 56,
                            "size of LSTM hidden memory")
tf.app.flags.DEFINE_integer('vd_amount', 28,
                            "vd_amount")
tf.app.flags.DEFINE_integer('rnn_layers', 1,
                            "number of stacked lstm")
tf.app.flags.DEFINE_integer('num_steps', 12,
                            "total steps of time")
tf.app.flags.DEFINE_boolean('is_float32', True,
                            "data type of the LSTM state, float32 if true, float16 otherwise")
tf.app.flags.DEFINE_float('learning_rate', 0,
                          "learning rate of RMSPropOptimizer")
tf.app.flags.DEFINE_float('decay_rate', 0,
                          "decay rate of RMSPropOptimizer")
tf.app.flags.DEFINE_float('momentum', 0,
                          "momentum of RMSPropOptimizer")


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
        self.vd_amount = FLAGS.vd_amount
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
        print("vd_amount:", self.vd_amount)
        print("rnn_layers:", self.rnn_layers)
        print("num_steps:", self.num_steps)
        print("is_float32:", self.is_float32)
        print("learning_rate:", self.learning_rate)
        print("decay_rate:", self.decay_rate)
        print("momentum:", self.momentum)


def main(_):
    with tf.get_default_graph().as_default() as graph:

        # read data [amount, num_steps, mileage, dfswt] == [None, 10, 28, 5]
        test_raw_data = np.load(FLAGS.data_dir + raw_data_name)
        test_label_data = np.load(FLAGS.data_dir + label_data_name)

        # select flow from [density, flow, speed, weekday, time]
        test_raw_data = test_raw_data[:, :, :, 1]
        temp = test_label_data[:, :, :]
        test_label_data = test_label_data[:, :, 1]

        # placeholder
        X_ph = tf.placeholder(dtype=tf.float32, shape=[
                              FLAGS.batch_size, FLAGS.num_steps, FLAGS.vd_amount], name='input_data')
        Y_ph = tf.placeholder(dtype=tf.float32, shape=[
                              FLAGS.batch_size, FLAGS.vd_amount], name='label_data')

        # config setting
        config = TestingConfig()
        config.show()

        # model
        model = model_lstm.TFPModel(config, is_training=True)
        logits_op = model.inference(inputs=X_ph)
        losses_op = model.losses(logits=logits_op, labels=Y_ph)
        mape_op = model.MAPE(logits=logits_op, labels=Y_ph)

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

            saver.restore(sess, FLAGS.checkpoints_dir + '-99')
            print("Successully restored!!")

            # testing
            tp1_result = None
            tp2_result = None
            next_test_raw_data = None
            test_loss_sum = 0.0
            test_mape_sum = 0.0
            flg = True
            for i in range(len(test_label_data) - 1):

                if temp[i][0][3] == 3:
                    flg = False

                if flg and temp[i][0][3] == 2:

                    if tp1_result is None or tp2_result is None:
                        current_X_batch = test_raw_data[i:i + 1]
                    else:
                        current_X_batch = next_test_raw_data

                    current_Y_batch = test_label_data[i:i + 1]
                    predicted_value, losses_value, mape_value = sess.run([logits_op, losses_op, mape_op], feed_dict={
                        X_ph: current_X_batch, Y_ph: current_Y_batch})
                    test_loss_sum += losses_value
                    test_mape_sum += mape_value

                    predicted_value_np = np.array(predicted_value)
                    predicted_value_np = np.reshape(
                        predicted_value_np, [1, 1, FLAGS.vd_amount])

                    tp1_result = tp2_result
                    tp2_result = predicted_value_np

                    if tp1_result is not None and tp2_result is not None:
                        next_test_raw_data = np.concatenate(
                            [test_raw_data[i + 1:i + 2, :-2, :], tp1_result, tp2_result], axis=1)

                    for vd_idx in range(FLAGS.vd_amount):
                        labels_scalar_summary = tf.Summary()
                        labels_scalar_summary.value.add(
                            simple_value=current_Y_batch[0][vd_idx], tag="cmp" + str(vd_idx))
                        labels_summary_writer.add_summary(
                            labels_scalar_summary, global_step=i)
                        labels_summary_writer.flush()

                        logits_scalar_summary = tf.Summary()
                        logits_scalar_summary.value.add(
                            simple_value=predicted_value[0][vd_idx], tag="cmp" + str(vd_idx))
                        logits_summary_writer.add_summary(
                            logits_scalar_summary, global_step=i)
                        logits_summary_writer.flush()

            # test mean loss
            test_mean_loss = test_loss_sum / len(test_label_data)
            test_mean_mape = test_mape_sum / len(test_label_data)

            print("testing mean loss: ", test_mean_loss)
            print("testing mean mape: ", test_mean_mape * 100.0, "%")

        # TODO: https://www.tensorflow.org/api_docs/python/tf/trai/Supervisor
        # sv = Supervisor(logdir=FLAGS.checkpoints_dir)
        # with sv.managed_session(FLAGS.master) as sess:
        #     while not sv.should_stop():
        #         sess.run(<my_train_op>)


if __name__ == "__main__":
    if not os.path.exists(FLAGS.checkpoints_dir):
        os.makedirs(FLAGS.checkpoints_dir)
    tf.app.run()
