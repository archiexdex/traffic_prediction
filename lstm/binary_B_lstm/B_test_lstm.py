from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import model_lstm
import datetime

raw_data_name = "batch_no_over_data_mile_15_28.5_total_60_predict_1_5.npy"
label_data_name = "label_no_over_data_mile_15_28.5_total_60_predict_1_5.npy"

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', '/home/nctucgv/Documents/TrafficVis_Run/src/traffic_flow_detection/',
                           "data directory")
tf.app.flags.DEFINE_string('checkpoints_dir', 'backlog_new/' + raw_data_name[6:-4] + '/checkpoints/',
                           "training checkpoints directory")
tf.app.flags.DEFINE_string('log_dir', 'backlog_new/' + raw_data_name[6:-4] + '/test_log_0/',
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
tf.app.flags.DEFINE_integer('day', 700,
                            "day")
tf.app.flags.DEFINE_integer('interval', 5,
                            "interval")


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

def the_date(day):
    return (datetime.date(2012,1,1) + datetime.timedelta(days=day-1) ).strftime("%Y-%m-%d")

def the_time(time):
    m = time % 60
    h = time / 60
    return time * 5 / 3

def main(_):
    with tf.get_default_graph().as_default() as graph:

        # read data [amount, num_steps, mileage, dfswt] == [None, 10, 28, 5]
        test_raw_data = np.load(FLAGS.data_dir + raw_data_name)
        test_label_data = np.load(FLAGS.data_dir + label_data_name)

        # select flow from [density, flow, speed, weekday, time, day]
        test_raw_data = test_raw_data[:, :, :, 1]
        test_label_all = test_label_data[:, 0:14, :]
        test_label_data = test_label_data[:, 0:14, 1]

        # placeholder
        X_ph = tf.placeholder(dtype=tf.float32, shape=[
                              None, FLAGS.num_steps, FLAGS.vd_amount], name='input_data')
        Y_ph = tf.placeholder(dtype=tf.float32, shape=[
                              None, FLAGS.vd_amount/2], name='label_data')

        # config setting
        config = TestingConfig()
        config.show()

        # model
        model = model_lstm.TFPModel(config, is_training=True)
        logits_op = model.inference(inputs=X_ph)
        losses_op = model.losses(logits=logits_op, labels=Y_ph)
        mape_op = model.MAPE(logits=logits_op, labels=Y_ph)

        init = tf.global_variables_initializer()
        # saver
        saver = tf.train.Saver()

        # np saver
        loss_saver = []

        i = 0
        # Session
        with tf.Session() as sess:
            sess.run(init)

            saver.restore(sess, FLAGS.checkpoints_dir + '-99')
            print("Successully restored!!")
            # for i, _ in enumerate(test_label_data):
            # while i < len(test_label_data) - FLAGS.batch_size:
            #     data  = test_raw_data[i:i+FLAGS.batch_size]
            #     label = test_label_data[i:i+FLAGS.batch_size]

            #     predicted_value, losses_value, mape_value = sess.run([logits_op, losses_op, mape_op], feed_dict={X_ph: data, Y_ph: label})
                
            #     print("ephoches: ", i, "trainng loss: ", losses_value)
            #     loss_saver.append(losses_value)
            #     i += FLAGS.batch_size
            # np.save("loss_lstm_"+raw_data_name, loss_saver)

            if FLAGS.day is None:
                pass
            else:
                # testing data of specific day
                # summary
                predict_loss_summary_writer = tf.summary.FileWriter(
                    FLAGS.log_dir + 'predicted_loss', graph=graph)
                # draw specific day
                amount_counter = 0
                for i, _ in enumerate(test_label_data):
                    if test_label_all[i][0][5] == FLAGS.day:
                        interval_id = 0
                        offset = i
                        while interval_id < (1440 // FLAGS.interval):
                            if test_label_all[offset][0][4] // FLAGS.interval != interval_id:
                                for vd_idx in range(FLAGS.vd_amount//2):
                                    predict_losses_scalar_summary = tf.Summary()
                                    predict_losses_scalar_summary.value.add(
                                        simple_value=0, tag="DAY:" + the_date(FLAGS.day) + "WEEK: " + str(test_label_all[i][0][3]) + " VD:" + str(vd_idx))
                                    predict_loss_summary_writer.add_summary(
                                        predict_losses_scalar_summary, global_step=the_time(interval_id * FLAGS.interval))
                                    predict_loss_summary_writer.flush()
                            else:
                                offset += 1
                                amount_counter += 1
                                current_X_batch = test_raw_data[offset:offset + 1]
                                current_Y_batch = test_label_data[offset:offset + 1]
                                predicted_value = sess.run(logits_op, feed_dict={
                                    X_ph: current_X_batch})

                                for vd_idx in range(FLAGS.vd_amount//2):
                                    predict_losses_scalar_summary = tf.Summary()
                                    predict_losses_scalar_summary.value.add(
                                        simple_value=predicted_value[0][vd_idx], tag="DAY:" + the_date(FLAGS.day) + "WEEK: " + str(test_label_all[i][0][3])+ " VD:" + str(vd_idx))
                                    predict_loss_summary_writer.add_summary(
                                        predict_losses_scalar_summary, global_step=the_time(interval_id * FLAGS.interval))
                                    predict_loss_summary_writer.flush()

                            interval_id += 1
                            if test_label_all[offset][0][4] < 100 and interval_id > 200:
                                break
 
                        print ("WEEK:", test_label_all[i][0][3])
                        break
                        

            #     # test mean loss
            #     test_mean_loss = test_loss_sum / amount_counter
            #     test_mean_mape = test_mape_sum / amount_counter

            #     print("testing mean loss: ", test_mean_loss)
            #     print("testing mean mape: ", test_mean_mape * 100.0, "%")

        # TODO: https://www.tensorflow.org/api_docs/python/tf/trai/Supervisor
        # sv = Supervisor(logdir=FLAGS.checkpoints_dir)
        # with sv.managed_session(FLAGS.master) as sess:
        #     while not sv.should_stop():
        #         sess.run(<my_train_op>)


if __name__ == "__main__":
    if not os.path.exists(FLAGS.checkpoints_dir):
        os.makedirs(FLAGS.checkpoints_dir)
    tf.app.run()
