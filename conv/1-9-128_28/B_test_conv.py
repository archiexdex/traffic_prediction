from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import model_conv
import plotly
plotly.tools.set_credentials_file(username='ChenChiehYu', api_key='tBPgjENSCbZ23SQ5qeFB')
import plotly.plotly as py
import plotly.graph_objs as go


raw_data_name = "batch_no_over_data_mile_15_28.5_total_60_predict_1_5.npy"
label_data_name = "label_no_over_data_mile_15_28.5_total_60_predict_1_5.npy"

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', '/home/nctucgv/Documents/TrafficVis_Run/src/traffic_flow_detection/',
                           "data directory")
tf.app.flags.DEFINE_string('checkpoints_dir', 'backlog_loss/' + raw_data_name[6:-4] + '/checkpoints/',
                           "training checkpoints directory")
tf.app.flags.DEFINE_string('log_dir', 'backlog_loss/' + raw_data_name[6:-4] + '/test_log_0/',
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
tf.app.flags.DEFINE_integer('interval', 5,
                            "interval")
tf.app.flags.DEFINE_integer('day', None,
                            "day")


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

        # read data [amount, total_interval, mileage, dfswt] == [None, 10, 28, 5]
        test_raw_data = np.load(FLAGS.data_dir + raw_data_name)
        test_label_data = np.load(FLAGS.data_dir + label_data_name)

        # select all from [density, flow, speed, weekday, time, day]
        test_raw_data = test_raw_data[:, :, :, :5]
        test_label_all = test_label_data[:, :, :]
        test_label_data = test_label_data[:, :, 1]

        # placeholder
        X_ph = tf.placeholder(dtype=tf.float32, shape=[
                              None, FLAGS.total_interval, FLAGS.vd_amount, 5], name='input_data')

        # config setting
        config = TestingConfig()
        config.show()

        # model
        model = model_conv.TFPModel(config, is_training=True)
        logits_op = model.inference(inputs=X_ph)

        init = tf.global_variables_initializer()
        # saver
        saver = tf.train.Saver()

        # Session
        with tf.Session() as sess:
            sess.run(init)

            saver.restore(sess, FLAGS.checkpoints_dir + '-99')
            print("Successully restored!!")

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
                                for vd_idx in range(FLAGS.vd_amount):
                                    predict_losses_scalar_summary = tf.Summary()
                                    predict_losses_scalar_summary.value.add(
                                        simple_value=0, tag="DAY:" + str(FLAGS.day) + " VD:" + str(vd_idx))
                                    predict_loss_summary_writer.add_summary(
                                        predict_losses_scalar_summary, global_step=interval_id * FLAGS.interval)
                                    predict_loss_summary_writer.flush()
                            else:
                                offset += 1
                                amount_counter += 1
                                current_X_batch = test_raw_data[offset:offset + 1]
                                current_Y_batch = test_label_data[offset:offset + 1]
                                predicted_value = sess.run(logits_op, feed_dict={
                                    X_ph: current_X_batch})

                                for vd_idx in range(FLAGS.vd_amount):
                                    predict_losses_scalar_summary = tf.Summary()
                                    predict_losses_scalar_summary.value.add(
                                        simple_value=predicted_value[0][vd_idx], tag="DAY:" + str(FLAGS.day) + " VD:" + str(vd_idx))
                                    predict_loss_summary_writer.add_summary(
                                        predict_losses_scalar_summary, global_step=interval_id * FLAGS.interval)
                                    predict_loss_summary_writer.flush()

                            interval_id += 1
                            if test_label_all[offset][0][4] < 100 and interval_id > 200:
                                break

                        print ("WEEK:", test_label_all[i][0][3])
                        break

        #     # testing all data
        #     predicted_value = sess.run([logits_op], feed_dict={X_ph: test_raw_data})
            
        #     predicted_value = np.array(predicted_value)
        #     print(predicted_value.shape)
        #     # Create a trace
        #     trace = go.Scatter(
        #         x = list(range(10)),
        #         y = predicted_value[0,:10]
        #     )

        #     data = [trace]
        #     py.plot(data, filename='basic-line')


if __name__ == "__main__":
    if not os.path.exists(FLAGS.checkpoints_dir):
        os.makedirs(FLAGS.checkpoints_dir)
    tf.app.run()
