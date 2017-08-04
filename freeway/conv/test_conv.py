"""
Training Data:
    input [num_data, total_intervals,   num_vds,    features]
        total_intervals: total number of intervals, 5 mins between every two data
        num_vds: number of vehicles detectors
        features: [density, flow, speed, weekday, time, day]
    label [num_data, target_intervals,  target_vds, target_features]
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import model_conv
import datetime
import time

FLAGS = tf.app.flags.FLAGS

# path parameters
tf.app.flags.DEFINE_string("raw_data", "batch_no_over_data_mile_15_28.5_total_60_predict_1_20.npy",
                           "raw data name")
tf.app.flags.DEFINE_string("label_data", "label_no_over_data_mile_15_28.5_total_60_predict_1_20.npy",
                           "label data name")
tf.app.flags.DEFINE_string('data_dir', '/home/nctucgv/Documents/TrafficVis_Run/src/traffic_flow_detection/',
                           "data directory")
tf.app.flags.DEFINE_string('restore_path', None,
                           "path of saving model eg: ../checkpoints/model.ckpt-5")
tf.app.flags.DEFINE_string('log_dir', 'predict_1_20/log/',
                           "summary directory")
# training parameters
tf.app.flags.DEFINE_integer('batch_size', 512,
                            "mini-batch size to test the whole testing data")
tf.app.flags.DEFINE_integer('vd_amount', 28,
                            "vd_amount")
tf.app.flags.DEFINE_integer('total_interval', 12,
                            "total intervals of time")
# target parameters
tf.app.flags.DEFINE_integer('target_vd', 12,
                            "number of vds to predict")
tf.app.flags.DEFINE_integer('target_interval', 4,
                            "number of interval to predict")
# drawing parameters
tf.app.flags.DEFINE_integer('day', 700,
                            "which specific day to draw")
tf.app.flags.DEFINE_integer('interval', 5,
                            "interval between every two data")
# B model
tf.app.flags.DEFINE_boolean('if_save_loss', False,
                            "if True, save losses as training data for B model, if False, go drawing predicted result")


class TestingConfig(object):
    """
    testing config
    """

    def __init__(self):
        self.data_dir = FLAGS.data_dir
        self.restore_path = FLAGS.restore_path
        self.log_dir = FLAGS.log_dir
        self.batch_size = FLAGS.batch_size
        self.vd_amount = FLAGS.vd_amount
        self.total_interval = FLAGS.total_interval

    def show(self):
        print("data_dir:", self.data_dir)
        print("restore_path:", self.restore_path)
        print("log_dir:", self.log_dir)
        print("batch_size:", self.batch_size)
        print("vd_amount:", self.vd_amount)
        print("total_interval:", self.total_interval)


def the_date(day):
    return (datetime.date(2012, 1, 1) + datetime.timedelta(days=day - 1)).strftime("%Y-%m-%d")


def the_time(time):
    return time * 5 / 3


def write_data(writer, value, week, vd_idx, time):
    """
    Params:
        writer: 
        value:
        week: 
        vd_idx:
        time:
    """
    summary = tf.Summary()
    summary.value.add(
        simple_value=value, tag="DAY:" + the_date(FLAGS.day) + "WEEK: " + str(week) + " VD:" + str(vd_idx))
    writer.add_summary(
        summary, the_time(time))
    # writer.flush()

def main(_):
    with tf.get_default_graph().as_default() as graph:
        # read data
        test_raw_data = np.load(FLAGS.data_dir + FLAGS.raw_data)
        test_label_data = np.load(FLAGS.data_dir + FLAGS.label_data)

        # select 5 training features from
        # [density, flow, speed, weekday, time, day]
        test_raw_data = test_raw_data[:, :, :, :5]
        # for drawing purpose
        test_label_all = test_label_data[:, :FLAGS.target_interval, :FLAGS.target_vd, :]
        # for loss calculation prupose, predict both flow and speed
        test_label_data = test_label_data[:,
                                          :FLAGS.target_interval, :FLAGS.target_vd, 1:3]

        # placeholder
        X_ph = tf.placeholder(dtype=tf.float32, shape=[
                              None, FLAGS.total_interval, FLAGS.vd_amount, 5], name='input_data')
        Y_ph = tf.placeholder(dtype=tf.float32, shape=[
                              None, FLAGS.target_interval, FLAGS.target_vd, 2], name='label_data')

        # config setting
        config = TestingConfig()
        config.show()

        # model
        model = model_conv.TFPModel(config, is_training=False)
        logits_op = model.inference(inputs=X_ph)
        losses_op = model.losses(logits=logits_op, labels=Y_ph)
        l2_losses_op = model.l2_losses(logits=logits_op, labels=Y_ph)

        init = tf.global_variables_initializer()
        # saver
        saver = tf.train.Saver()

        # Session
        with tf.Session() as sess:
            sess.run(init)

            # restore model if exist
            if FLAGS.restore_path is not None:
                saver.restore(sess, FLAGS.restore_path)
                print("Model restored:", FLAGS.restore_path)

            # save loss for b model
            if FLAGS.if_save_loss:
                losses_value_all = []
                for i in range(len(test_raw_data) // 512):
                    predicted_value, losses_value = sess.run([logits_op, l2_losses_op], feed_dict={
                        X_ph: test_raw_data[i * 512:i * 512 + 512], Y_ph: test_label_data[i * 512:i * 512 + 512]})
                    losses_value_all.append(losses_value)
                losses_value_all = np.concatenate(losses_value_all, axis=0)
                np.save("loss_" + raw_data_name, losses_value_all)
                print("save loss.. successfully")

            # evaluate the testing loss for whole testing set
            if FLAGS.day is None:
                losses_value_all = 0
                for i in range(len(test_raw_data) // 512):
                    predicted_value, losses_value = sess.run([logits_op, losses_op], feed_dict={
                        X_ph: test_raw_data[i * 512:i * 512 + 512], Y_ph: test_label_data[i * 512:i * 512 + 512]})
                    losses_value_all += losses_value

                print("testing mean loss: ", losses_value_all /
                      (len(test_raw_data) // 512))
            # draw the prediction of specific day on tensorboard
            else:

                test_loss_sum = 0.0
                amount_counter = 0

                # find the first index which [its day] is equal to [target day] of the
                # whole input data
                interval_id = 0
                # offset = np.argmax(test_label_all[:, :, :, 5] == FLAGS.day)
                # offset %= (test_label_all.shape[0])
                offset = np.argwhere(test_label_all[:, :, :, 5] == FLAGS.day)[0][0]
                print(offset)
                print(test_label_all.shape[0])
                week_day = test_label_all[offset][0][0][3]
                while interval_id < (60 * 24 // FLAGS.interval):
                    # if missing data, record as 0
                    if test_label_all[offset][0][0][4] // FLAGS.interval != interval_id:
                        interval_id += 1
                        for i in range(FLAGS.target_interval):
                            target_interval_range = str((i + 1) * FLAGS.interval)
                            # summary
                            density_summary_writer = tf.summary.FileWriter(
                                FLAGS.log_dir + target_interval_range + '_density')
                            flow_summary_writer = tf.summary.FileWriter(
                                FLAGS.log_dir + target_interval_range + '_flow')
                            speed_summary_writer = tf.summary.FileWriter(
                                FLAGS.log_dir + target_interval_range + '_speed')

                            predict_flow_summary_writer = tf.summary.FileWriter(
                                FLAGS.log_dir + target_interval_range + '_prediction_flow')
                            predict_speed_summary_writer = tf.summary.FileWriter(
                                FLAGS.log_dir + target_interval_range + '_prediction_speed')
                            losses_summary_writer = tf.summary.FileWriter(
                                FLAGS.log_dir + target_interval_range + '_l2_losses')

                            for vd_idx in range(FLAGS.target_vd):
                                write_data(
                                    density_summary_writer, 0, week_day, vd_idx, interval_id * FLAGS.interval)
                                write_data(
                                    flow_summary_writer, 0, week_day, vd_idx, interval_id * FLAGS.interval)
                                write_data(
                                    speed_summary_writer, 0, week_day, vd_idx, interval_id * FLAGS.interval)

                                write_data(
                                    predict_flow_summary_writer, 0, week_day, vd_idx, interval_id * FLAGS.interval)
                                write_data(
                                    predict_speed_summary_writer, 0, week_day, vd_idx, interval_id * FLAGS.interval)
                                write_data(
                                    losses_summary_writer, 0, week_day, vd_idx, interval_id * FLAGS.interval)
                            density_summary_writer.close()
                            flow_summary_writer.close()
                            speed_summary_writer.close()
                            predict_flow_summary_writer.close()
                            predict_speed_summary_writer.close()
                            losses_summary_writer.close()
                            time.sleep(0.25)
                    else:
                        offset += 1
                        interval_id += 1
                        amount_counter += 1
                        current_X_batch = test_raw_data[offset:offset + 1]
                        current_Y_batch = test_label_data[offset:offset + 1]
                        predicted_value, losses_value = sess.run([logits_op, losses_op], feed_dict={
                            X_ph: current_X_batch, Y_ph: current_Y_batch})
                        test_loss_sum += losses_value
                        for i in range(FLAGS.target_interval):
                            target_interval_range = str((i + 1) * FLAGS.interval)

                            # summary
                            density_summary_writer = tf.summary.FileWriter(
                                FLAGS.log_dir + target_interval_range + '_density')
                            flow_summary_writer = tf.summary.FileWriter(
                                FLAGS.log_dir + target_interval_range + '_flow')
                            speed_summary_writer = tf.summary.FileWriter(
                                FLAGS.log_dir + target_interval_range + '_speed')

                            predict_flow_summary_writer = tf.summary.FileWriter(
                                FLAGS.log_dir + target_interval_range + '_prediction_flow')
                            predict_speed_summary_writer = tf.summary.FileWriter(
                                FLAGS.log_dir + target_interval_range + '_prediction_speed')
                            losses_summary_writer = tf.summary.FileWriter(
                                FLAGS.log_dir + target_interval_range + '_l2_losses')

                            for vd_idx in range(FLAGS.target_vd):

                                write_data(
                                    density_summary_writer, test_label_all[offset][i][vd_idx][0], week_day, vd_idx, interval_id * FLAGS.interval)
                                write_data(
                                    flow_summary_writer, test_label_all[offset][i][vd_idx][1], week_day, vd_idx, interval_id * FLAGS.interval)
                                write_data(
                                    speed_summary_writer, test_label_all[offset][i][vd_idx][2], week_day, vd_idx, interval_id * FLAGS.interval)

                                write_data(
                                    predict_flow_summary_writer, predicted_value[0][i][vd_idx][0], week_day, vd_idx, interval_id * FLAGS.interval)
                                write_data(
                                    predict_speed_summary_writer, predicted_value[0][i][vd_idx][1], week_day, vd_idx, interval_id * FLAGS.interval)
                                write_data(
                                    losses_summary_writer, losses_value, week_day, vd_idx, interval_id * FLAGS.interval)

                            density_summary_writer.close()
                            flow_summary_writer.close()
                            speed_summary_writer.close()
                            predict_flow_summary_writer.close()
                            predict_speed_summary_writer.close()
                            losses_summary_writer.close()
                            time.sleep(0.25)

                    # if test_label_all[offset][0][4] < 100 and interval_id > 200:
                    #     break

                print ("WEEK:", test_label_all[offset][0][0][3])

                # test mean loss
                test_mean_loss = test_loss_sum / amount_counter

                print("testing mean loss: ", test_mean_loss)


if __name__ == "__main__":
    tf.app.run()
