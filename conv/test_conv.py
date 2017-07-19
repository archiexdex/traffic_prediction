from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import model_conv
import datetime

raw_data_name = "batch_no_over_data_mile_15_28.5_total_60_predict_1_20.npy"
label_data_name = "label_no_over_data_mile_15_28.5_total_60_predict_1_20.npy"

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', '/home/nctucgv/Documents/TrafficVis_Run/src/traffic_flow_detection/',
                           "data directory")
tf.app.flags.DEFINE_string('checkpoints_dir', 'backlog_new/' + "predict_1_20" + '/checkpoints/',
                           "training checkpoints directory")
tf.app.flags.DEFINE_string('log_dir', 'backlog_new/' + "predict_1_20" + '/log/test-20/',
                           "summary directory")
tf.app.flags.DEFINE_integer('batch_size', 1,
                            "mini-batch size")
tf.app.flags.DEFINE_integer('total_epoches', 0,
                            "total training epoches")
tf.app.flags.DEFINE_integer('vd_amount', 28,
                            "vd_amount")
tf.app.flags.DEFINE_integer('total_interval', 12,
                            "total steps of time")
tf.app.flags.DEFINE_float('learning_rate', 0,
                          "learning rate of RMSPropOptimizer")
tf.app.flags.DEFINE_float('decay_rate', 0,
                          "decay rate of RMSPropOptimizer")
tf.app.flags.DEFINE_float('momentum', 0,
                          "momentum of RMSPropOptimizer")
tf.app.flags.DEFINE_integer('day', 700,
                            "day")
tf.app.flags.DEFINE_boolean('if_save_loss', False,
                            "save lossas training set for B model")
tf.app.flags.DEFINE_integer('interval', 5,
                            "interval")
tf.app.flags.DEFINE_integer('interval_id', 3,
                            "0->5, 1->10, 2->15, 3->20")


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
        self.vd_amount = FLAGS.vd_amount
        self.total_interval = FLAGS.total_interval
        self.learning_rate = FLAGS.learning_rate
        self.decay_rate = FLAGS.decay_rate
        self.momentum = FLAGS.momentum

    def show(self):
        print("data_dir:", self.data_dir)
        print("checkpoints_dir:", self.checkpoints_dir)
        print("log_dir:", self.log_dir)
        print("batch_size:", self.batch_size)
        print("total_epoches:", self.total_epoches)
        print("vd_amount:", self.vd_amount)
        print("total_interval:", self.total_interval)
        print("learning_rate:", self.learning_rate)
        print("decay_rate:", self.decay_rate)
        print("momentum:", self.momentum)


def the_date(day):
    return (datetime.date(2012, 1, 1) + datetime.timedelta(days=day - 1)).strftime("%Y-%m-%d")


def the_time(time):
    m = time % 60
    h = time / 60
    return time * 5 / 3


def write_data(writer, value, week, vd_idx, time):
    summary = tf.Summary()
    summary.value.add(
        simple_value=value, tag="DAY:" + the_date(FLAGS.day) + "WEEK: " + str(week) + " VD:" + str(vd_idx))
    writer.add_summary(
        summary, the_time(time))
    writer.flush()


def main(_):
    with tf.get_default_graph().as_default() as graph:

        # np saver
        raw_loss = []

        # read data [amount, total_interval, mileage, dfswt] == [None, 10, 28,
        # 5]
        test_raw_data = np.load(FLAGS.data_dir + raw_data_name)
        test_label_data = np.load(FLAGS.data_dir + label_data_name)
        #TODO
        # select all from [density, flow, speed, weekday, time, day]
        test_raw_data = test_raw_data[:, :, :, :5]
        test_label_all = test_label_data[:, FLAGS.interval_id, 0:14, :]
        print(test_label_all.shape)
        test_label_data = test_label_data[:, :, 0:14, 1:1 + 2]

        # placeholder
        X_ph = tf.placeholder(dtype=tf.float32, shape=[
                              FLAGS.batch_size, FLAGS.total_interval, FLAGS.vd_amount, 5], name='input_data')
        Y_ph = tf.placeholder(dtype=tf.float32, shape=[
                              FLAGS.batch_size, 4, FLAGS.vd_amount / 2, 2], name='label_data')

        # config setting
        config = TestingConfig()
        config.show()

        # model
        model = model_conv.TFPModel(config, is_training=True)
        logits_op = model.inference(inputs=X_ph)
        losses_op = model.losses(logits=logits_op, labels=Y_ph)
        l2_losses_op = model.l2_losses(logits=logits_op, labels=Y_ph)
        mape_op = model.MAPE(logits=logits_op, labels=Y_ph)

        init = tf.global_variables_initializer()
        # saver
        saver = tf.train.Saver()

        # Session
        with tf.Session() as sess:
            sess.run(init)

            saver.restore(sess, FLAGS.checkpoints_dir + '-999')
            print("Successully restored!!")

            if FLAGS.if_save_loss:
                losses_value_all = []
                for i in range(len(test_raw_data) // 512):
                    predicted_value, losses_value = sess.run([logits_op, l2_losses_op], feed_dict={
                        X_ph: test_raw_data[i * 512:i * 512 + 512], Y_ph: test_label_data[i * 512:i * 512 + 512]})
                    losses_value_all.append(losses_value)
                losses_value_all = np.concatenate(losses_value_all, axis=0)
                print(losses_value_all.shape)
                # exit()
                np.save("loss_" + raw_data_name, losses_value_all)
                print("save loss.. successful XD")

            if FLAGS.day is None:
                # testing all data
                losses_value_all = 0
                for i in range(len(test_raw_data) // 512):
                    predicted_value, losses_value = sess.run([logits_op, losses_op], feed_dict={
                        X_ph: test_raw_data[i * 512:i * 512 + 512], Y_ph: test_label_data[i * 512:i * 512 + 512]})
                    losses_value_all += losses_value

                print("testing mean loss: ", losses_value_all /
                      (len(test_raw_data) // 512))
            else:
                # testing data of specific day
                # summary
                density_summary_writer = tf.summary.FileWriter(
                    FLAGS.log_dir + 'density')
                flow_summary_writer = tf.summary.FileWriter(
                    FLAGS.log_dir + 'flow')
                speed_summary_writer = tf.summary.FileWriter(
                    FLAGS.log_dir + 'speed')

                predict_flow_summary_writer = tf.summary.FileWriter(
                    FLAGS.log_dir + 'prediction_flow')
                predict_speed_summary_writer = tf.summary.FileWriter(
                    FLAGS.log_dir + 'prediction_speed')
                losses_summary_writer = tf.summary.FileWriter(
                    FLAGS.log_dir + 'l2_losses')

                # draw specific day
                test_loss_sum = 0.0
                test_mape_sum = 0.0
                amount_counter = 0
                for i, _ in enumerate(test_label_data):
                    if test_label_all[i][0][5] == FLAGS.day:

                        interval_id = 0
                        offset = i
                        while interval_id < (1440 // FLAGS.interval):
                            if test_label_all[offset][0][4] // FLAGS.interval != interval_id:
                                for vd_idx in range(FLAGS.vd_amount // 2):

                                    write_data(
                                        density_summary_writer, 0, test_label_all[i][0][3], vd_idx, interval_id * FLAGS.interval)
                                    write_data(
                                        flow_summary_writer, 0, test_label_all[i][0][3], vd_idx, interval_id * FLAGS.interval)
                                    write_data(
                                        speed_summary_writer, 0, test_label_all[i][0][3], vd_idx, interval_id * FLAGS.interval)

                                    write_data(
                                        predict_flow_summary_writer, 0, test_label_all[i][0][3], vd_idx, interval_id * FLAGS.interval)
                                    write_data(
                                        predict_speed_summary_writer, 0, test_label_all[i][0][3], vd_idx, interval_id * FLAGS.interval)
                                    write_data(
                                        losses_summary_writer, 0, test_label_all[i][0][3], vd_idx, interval_id * FLAGS.interval)

                            else:
                                offset += 1
                                amount_counter += 1
                                current_X_batch = test_raw_data[offset:offset + 1]
                                current_Y_batch = test_label_data[offset:offset + 1]
                                l2_losses_value, predicted_value, losses_value, mape_value = sess.run([l2_losses_op, logits_op, losses_op, mape_op], feed_dict={
                                    X_ph: current_X_batch, Y_ph: current_Y_batch})
                                #TODO
                                predicted_value = predicted_value[:, FLAGS.interval_id]
                                test_loss_sum += losses_value
                                test_mape_sum += mape_value

                                for vd_idx in range(FLAGS.vd_amount // 2):

                                    write_data(
                                        density_summary_writer, test_label_all[offset][vd_idx][0], test_label_all[i][0][3], vd_idx, interval_id * FLAGS.interval)
                                    write_data(
                                        flow_summary_writer, test_label_all[offset][vd_idx][1], test_label_all[i][0][3], vd_idx, interval_id * FLAGS.interval)
                                    write_data(
                                        speed_summary_writer, test_label_all[offset][vd_idx][2], test_label_all[i][0][3], vd_idx, interval_id * FLAGS.interval)

                                    write_data(
                                        predict_flow_summary_writer, predicted_value[0][vd_idx][0], test_label_all[i][0][3], vd_idx, interval_id * FLAGS.interval)
                                    write_data(
                                        predict_speed_summary_writer, predicted_value[0][vd_idx][1], test_label_all[i][0][3], vd_idx, interval_id * FLAGS.interval)
                                    write_data(losses_summary_writer, losses_value,
                                               test_label_all[i][0][3], vd_idx, interval_id * FLAGS.interval)

                            interval_id += 1
                            if test_label_all[offset][0][4] < 100 and interval_id > 200:
                                break

                        print ("WEEK:", test_label_all[i][0][3])
                        break

                # test mean loss
                test_mean_loss = test_loss_sum / amount_counter
                test_mean_mape = test_mape_sum / amount_counter

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
