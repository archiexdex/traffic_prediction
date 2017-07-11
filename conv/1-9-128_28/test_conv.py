from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import model_conv

raw_data_name = "batch_no_over_data_mile_15_28.5_total_60_predict_1_5.npy"
label_data_name = "label_no_over_data_mile_15_28.5_total_60_predict_1_5.npy"

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', '/home/nctucgv/Documents/TrafficVis_Run/src/traffic_flow_detection/',
                           "data directory")
tf.app.flags.DEFINE_string('checkpoints_dir', 'backlog_new_new/' + raw_data_name[6:-4] + '/checkpoints/',
                           "training checkpoints directory")
tf.app.flags.DEFINE_string('log_dir', 'backlog_new_new/' + raw_data_name[6:-4] + '/test_log_0/',
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
tf.app.flags.DEFINE_integer('day', None,
                            "day")
tf.app.flags.DEFINE_boolean('if_save_loss', False,
                            "save lossas training set for B model")
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


def main(_):
    with tf.get_default_graph().as_default() as graph:

        # np saver
        raw_loss = []

        # read data [amount, total_interval, mileage, dfswt] == [None, 10, 28,
        # 5]
        test_raw_data = np.load(FLAGS.data_dir + raw_data_name)
        test_label_data = np.load(FLAGS.data_dir + label_data_name)

        # select all from [density, flow, speed, weekday, time, day]
        test_raw_data = test_raw_data[:, :, :, :5]
        test_label_all = test_label_data[:, :, :]
        test_label_data = test_label_data[:, :, 1]

        # placeholder
        X_ph = tf.placeholder(dtype=tf.float32, shape=[
                              None, FLAGS.total_interval, FLAGS.vd_amount, 5], name='input_data')
        Y_ph = tf.placeholder(dtype=tf.float32, shape=[
                              None, FLAGS.vd_amount], name='label_data')

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

            saver.restore(sess, FLAGS.checkpoints_dir + '-99')
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
                labels_summary_writer = tf.summary.FileWriter(
                    FLAGS.log_dir + 'observation', graph=graph)
                logits_summary_writer = tf.summary.FileWriter(
                    FLAGS.log_dir + 'prediction', graph=graph)
                losses_summary_writer = tf.summary.FileWriter(
                    FLAGS.log_dir + 'l2_losses', graph=graph)
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
                                for vd_idx in range(FLAGS.vd_amount):
                                    labels_scalar_summary = tf.Summary()
                                    labels_scalar_summary.value.add(
                                        simple_value=0, tag="DAY:" + str(FLAGS.day) + " VD:" + str(vd_idx))
                                    labels_summary_writer.add_summary(
                                        labels_scalar_summary, global_step=interval_id * FLAGS.interval)
                                    labels_summary_writer.flush()

                                    logits_scalar_summary = tf.Summary()
                                    logits_scalar_summary.value.add(
                                        simple_value=0, tag="DAY:" + str(FLAGS.day) + " VD:" + str(vd_idx))
                                    logits_summary_writer.add_summary(
                                        logits_scalar_summary, global_step=interval_id * FLAGS.interval)
                                    logits_summary_writer.flush()

                                    losses_scalar_summary = tf.Summary()
                                    losses_scalar_summary.value.add(
                                        simple_value=0, tag="DAY:" + str(FLAGS.day) + " VD:" + str(vd_idx))
                                    losses_summary_writer.add_summary(
                                        losses_scalar_summary, global_step=interval_id * FLAGS.interval)
                                    losses_summary_writer.flush()
                            else:
                                offset += 1
                                amount_counter += 1
                                current_X_batch = test_raw_data[offset:offset + 1]
                                current_Y_batch = test_label_data[offset:offset + 1]
                                l2_losses_value, predicted_value, losses_value, mape_value = sess.run([l2_losses_op, logits_op, losses_op, mape_op], feed_dict={
                                    X_ph: current_X_batch, Y_ph: current_Y_batch})
                                test_loss_sum += losses_value
                                test_mape_sum += mape_value

                                for vd_idx in range(FLAGS.vd_amount):
                                    labels_scalar_summary = tf.Summary()
                                    labels_scalar_summary.value.add(
                                        simple_value=current_Y_batch[0][vd_idx], tag="DAY:" + str(FLAGS.day) + " VD:" + str(vd_idx))
                                    labels_summary_writer.add_summary(
                                        labels_scalar_summary, global_step=interval_id * FLAGS.interval)
                                    labels_summary_writer.flush()

                                    logits_scalar_summary = tf.Summary()
                                    logits_scalar_summary.value.add(
                                        simple_value=predicted_value[0][vd_idx], tag="DAY:" + str(FLAGS.day) + " VD:" + str(vd_idx))
                                    logits_summary_writer.add_summary(
                                        logits_scalar_summary, global_step=interval_id * FLAGS.interval)
                                    logits_summary_writer.flush()

                                    losses_scalar_summary = tf.Summary()
                                    losses_scalar_summary.value.add(
                                        simple_value=l2_losses_value[0][vd_idx], tag="DAY:" + str(FLAGS.day) + " VD:" + str(vd_idx))
                                    losses_summary_writer.add_summary(
                                        losses_scalar_summary, global_step=interval_id * FLAGS.interval)
                                    losses_summary_writer.flush()

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
