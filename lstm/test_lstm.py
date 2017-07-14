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
tf.app.flags.DEFINE_string('log_dir', 'backlog_new/' + raw_data_name[6:-4] + '/test_log_5/',
                           "summary directory")
tf.app.flags.DEFINE_integer('batch_size', 512,
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
        Z_ph = tf.placeholder(dtype=tf.float32, shape=[
                              None, FLAGS.vd_amount], name='shift_input_data')
            

        # config setting
        config = TestingConfig()
        config.show()
        
        # model
        model = model_lstm.TFPModel(config, is_training=True)
        logits_op = model.inference(inputs=X_ph)
        losses_op = model.losses(logits=logits_op, labels=Y_ph)
        l1_loss_op = model.losses(logits=logits_op, labels=Y_ph, is_squared=False, is_reduction=True)
        l1_losses_op = model.losses(logits=logits_op, labels=Y_ph, is_squared=False, is_reduction=False)

        l2_loss_op = model.losses(logits=logits_op, labels=Y_ph, is_squared=True, is_reduction=True)
        l2_losses_op = model.losses(logits=logits_op, labels=Y_ph, is_squared=True, is_reduction=False)

        mape_op = model.MAPE(logits=logits_op, labels=Y_ph)

        init = tf.global_variables_initializer()
        
        # saver
        saver = tf.train.Saver()
        
        # Session
        with tf.Session() as sess:
            sess.run(init)
            
            saver.restore(sess, FLAGS.checkpoints_dir + '-99')
            print("Successully restored!!")
            
            
            special_points = []
            losses_value_all = []
            i = 0
            while i < len(test_label_data) - FLAGS.batch_size:
                
                data  = test_raw_data[i:i+FLAGS.batch_size]
                label = test_label_data[i:i+FLAGS.batch_size]

                predicted_value, losses_value = sess.run([logits_op, l1_losses_op], feed_dict={
                    X_ph: data, Y_ph: label})
                
                # for ptr, value in enumerate(losses_value):
                #     for item in value:
                #         if item > 400:
                #             special_points.append(ptr+i)
                #             break

                # print("ephoches: ", i, "trainng loss: ", losses_value)
                
                losses_value_all.append(losses_value)
                i += FLAGS.batch_size
            print("save loss.. successful XD")

            print(np.array(losses_value_all).shape)
            losses_value_all = np.concatenate(losses_value_all,axis=0)
            np.save("loss_lstm_"+raw_data_name, losses_value_all)
            
            # special_day = []
            # for ptr in test_label_all:
            #     # speed
            #     for vd in ptr:
            #         if vd[2] < 20 and ( the_time(vd[4]) > 800 and the_time(vd[4]) < 2400 ):
            #             if len(special_day) > 0 and special_day[-1] == vd[5]:
            #                 continue
            #             special_day.append(vd[5])
            #             break
            # np.save("special_day",special_day)


            if FLAGS.day is None:
                # testing all data
                predicted_value, losses_value, mape_value = sess.run([logits_op, losses_op, mape_op], feed_dict={
                    X_ph: test_raw_data, Y_ph: test_label_data})

                print("testing mean loss: ", losses_value)
                print("testing mean mape: ", mape_value * 100.0, "%")
            else:
                # summary
                labels_summary_writer = tf.summary.FileWriter(
                    FLAGS.log_dir + 'observation')
                logits_summary_writer = tf.summary.FileWriter(
                    FLAGS.log_dir + 'prediction')
                loss_summary_writer = tf.summary.FileWriter(
                    FLAGS.log_dir + 'loss')
                shift_summary_writer = tf.summary.FileWriter(
                    FLAGS.log_dir + 'observation_shift')
                speed_summary_writer = tf.summary.FileWriter(
                    FLAGS.log_dir + 'speed')

                # draw specific day
                test_loss_sum = 0.0
                test_mape_sum = 0.0
                amount_counter = 0
                
                for i, _ in enumerate(test_label_data):
                    if test_label_all[i][0][5] == FLAGS.day:
                        interval_id = 0
                        offset = i
                        while interval_id < (1440//FLAGS.interval) :
                            if test_label_all[offset][0][4]//FLAGS.interval != interval_id:
                                for vd_idx in range(FLAGS.vd_amount//2):
                                    labels_scalar_summary = tf.Summary()
                                    labels_scalar_summary.value.add(
                                        simple_value=0, tag="DAY:" + the_date(FLAGS.day) + "WEEK: " + str(test_label_all[i][0][3]) + " VD:" + str(vd_idx))
                                    labels_summary_writer.add_summary(
                                        labels_scalar_summary, the_time(interval_id*FLAGS.interval))
                                    labels_summary_writer.flush()

                                    logits_scalar_summary = tf.Summary()
                                    logits_scalar_summary.value.add(
                                        simple_value=0, tag="DAY:" + the_date(FLAGS.day) + "WEEK: " + str(test_label_all[i][0][3]) + " VD:" + str(vd_idx))
                                    logits_summary_writer.add_summary(
                                        logits_scalar_summary, the_time(interval_id*FLAGS.interval))
                                    logits_summary_writer.flush()

                                    shift_scalar_summary = tf.Summary()
                                    shift_scalar_summary.value.add(
                                        simple_value=0, tag="DAY:" + the_date(FLAGS.day) + "WEEK: " + str(test_label_all[i][0][3]) + " VD:" + str(vd_idx))
                                    shift_summary_writer.add_summary(
                                        shift_scalar_summary, the_time(interval_id*FLAGS.interval))
                                    shift_summary_writer.flush()

                                    loss_scalar_summary = tf.Summary()
                                    loss_scalar_summary.value.add(
                                        simple_value=0, tag="DAY:" + the_date(FLAGS.day) + "WEEK: " + str(test_label_all[i][0][3]) + " VD:" + str(vd_idx))
                                    loss_summary_writer.add_summary(
                                        loss_scalar_summary, the_time(interval_id*FLAGS.interval))
                                    loss_summary_writer.flush()   

                                    speed_scalar_summary = tf.Summary()
                                    speed_scalar_summary.value.add(
                                        simple_value=0, tag="DAY:" + the_date(FLAGS.day) + "WEEK: " + str(test_label_all[i][0][3]) + " VD:" + str(vd_idx))
                                    speed_summary_writer.add_summary(
                                        speed_scalar_summary, the_time(interval_id*FLAGS.interval))
                                    speed_summary_writer.flush()
                            else:
                                offset += 1
                                amount_counter += 1
                                current_X_batch = test_raw_data[offset:offset + 1]
                                current_Y_batch = test_label_data[offset:offset + 1]
                                predicted_value, losses_value, mape_value = sess.run([logits_op, l1_losses_op, mape_op], feed_dict={
                                    X_ph: current_X_batch, Y_ph: current_Y_batch})
                                test_loss_sum += losses_value
                                test_mape_sum += mape_value

                                for vd_idx in range(FLAGS.vd_amount//2):
                                    labels_scalar_summary = tf.Summary()
                                    labels_scalar_summary.value.add(
                                        simple_value=current_Y_batch[0][vd_idx], tag="DAY:" + the_date(FLAGS.day) + "WEEK: " + str(test_label_all[i][0][3]) + " VD:" + str(vd_idx))
                                    labels_summary_writer.add_summary(
                                        labels_scalar_summary, the_time(interval_id*FLAGS.interval))
                                    labels_summary_writer.flush()

                                    logits_scalar_summary = tf.Summary()
                                    logits_scalar_summary.value.add(
                                        simple_value=predicted_value[0][vd_idx], tag="DAY:" + the_date(FLAGS.day) + "WEEK: " + str(test_label_all[i][0][3]) + " VD:" + str(vd_idx))
                                    logits_summary_writer.add_summary(
                                        logits_scalar_summary, the_time(interval_id*FLAGS.interval))
                                    logits_summary_writer.flush()

                                    shift_scalar_summary = tf.Summary()
                                    shift_scalar_summary.value.add(
                                        simple_value=current_X_batch[0][-1][vd_idx], tag="DAY:" + the_date(FLAGS.day) + "WEEK: " + str(test_label_all[i][0][3]) + " VD:" + str(vd_idx))
                                    shift_summary_writer.add_summary(
                                        shift_scalar_summary, the_time(interval_id*FLAGS.interval))
                                    shift_summary_writer.flush()

                                    loss_scalar_summary = tf.Summary()
                                    loss_scalar_summary.value.add(
                                        simple_value=losses_value[0][vd_idx], tag="DAY:" + the_date(FLAGS.day) + "WEEK: " + str(test_label_all[i][0][3]) + " VD:" + str(vd_idx))
                                    loss_summary_writer.add_summary(
                                        loss_scalar_summary, the_time(interval_id*FLAGS.interval))
                                    loss_summary_writer.flush()
                                    
                                    speed_scalar_summary = tf.Summary()
                                    speed_scalar_summary.value.add(
                                        simple_value=test_label_all[offset][vd_idx][2], tag="DAY:" + the_date(FLAGS.day) + "WEEK: " + str(test_label_all[i][0][3]) + " VD:" + str(vd_idx))
                                    speed_summary_writer.add_summary(
                                        speed_scalar_summary, the_time(interval_id*FLAGS.interval))
                                    speed_summary_writer.flush()

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
