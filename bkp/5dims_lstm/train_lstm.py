from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import model_lstm


raw_data_name = "batch_no_over_data_mile_15_28.5_total_60_predict_1_5.npy"
label_data_name = "label_no_over_data_mile_15_28.5_total_60_predict_1_5.npy"

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', '/home/nctucgv/Documents/TrafficVis_Run/src/traffic_flow_detection/',
                           "data directory")
tf.app.flags.DEFINE_string('checkpoints_dir', 'backlog/5dims_hunit140_' + raw_data_name[6:-4] + '/checkpoints/',
                           "training checkpoints directory")
tf.app.flags.DEFINE_string('log_dir', 'backlog/5dims_hunit140_' + raw_data_name[6:-4] + '/log/',
                           "summary directory")
tf.app.flags.DEFINE_integer('batch_size', 512,
                            "mini-batch size")
tf.app.flags.DEFINE_integer('total_epoches', 100,
                            "total training epoches")
tf.app.flags.DEFINE_integer('hidden_size', 28*5,
                            "size of LSTM hidden memory")
tf.app.flags.DEFINE_integer('vd_amount', 28,
                            "vd_amount")
tf.app.flags.DEFINE_integer('rnn_layers', 1,
                            "number of stacked lstm")
tf.app.flags.DEFINE_integer('num_steps', 12,
                            "total steps of time")
tf.app.flags.DEFINE_boolean('is_float32', True,
                            "data type of the LSTM state, float32 if true, float16 otherwise")
tf.app.flags.DEFINE_float('learning_rate', 0.0001,
                          "learning rate of RMSPropOptimizer")
tf.app.flags.DEFINE_float('decay_rate', 0.99,
                          "decay rate of RMSPropOptimizer")
tf.app.flags.DEFINE_float('momentum', 0.9,
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
        global_steps = tf.train.get_or_create_global_step(graph=graph)

        # read data
        raw_data_t = np.load(FLAGS.data_dir + raw_data_name)
        label_data_t = np.load(FLAGS.data_dir + label_data_name)

        # select flow from [density, flow, speed, weekday, time]
        label_data_t = label_data_t[:, :, 1]

        # concat for later shuffle
        concat = np.c_[raw_data_t.reshape(len(raw_data_t), -1),
                       label_data_t.reshape(len(label_data_t), -1)]
        raw_data = concat[:, :raw_data_t.size //
                          len(raw_data_t)].reshape(raw_data_t.shape)
        label_data = concat[:, raw_data_t.size //
                            len(raw_data_t):].reshape(label_data_t.shape)
        del raw_data_t
        del label_data_t

        np.random.shuffle(concat)

        train_raw_data_t, test_raw_data = np.split(
            raw_data, [raw_data.shape[0] * 8 // 9])
        train_label_data_t, test_label_data = np.split(
            label_data, [label_data.shape[0] * 8 // 9])

        # concat for later shuffle
        concat = np.c_[train_raw_data_t.reshape(len(train_raw_data_t), -1),
                       train_label_data_t.reshape(len(train_label_data_t), -1)]
        train_raw_data = concat[:, :train_raw_data_t.size //
                                len(train_raw_data_t)].reshape(train_raw_data_t.shape)
        train_label_data = concat[:, train_raw_data_t.size //
                                  len(train_raw_data_t):].reshape(train_label_data_t.shape)
        del train_raw_data_t
        del train_label_data_t

        # placeholder
        X_ph = tf.placeholder(dtype=tf.float32, shape=[
                              FLAGS.batch_size, FLAGS.num_steps, FLAGS.vd_amount, 5], name='input_data')
        Y_ph = tf.placeholder(dtype=tf.float32, shape=[
                              FLAGS.batch_size, FLAGS.vd_amount], name='label_data')

        # config setting
        config = TestingConfig()
        config.show()

        # model
        model = model_lstm.TFPModel(config, is_training=True)
        logits_op = model.inference(inputs=X_ph)
        loss_op = model.losses(logits=logits_op, labels=Y_ph)
        train_op = model.train(loss=loss_op, global_step=global_steps)
        mape_op = model.MAPE(logits=logits_op, labels=Y_ph)

        # summary
        merged_op = tf.summary.merge_all()
        train_summary_writer = tf.summary.FileWriter(
            FLAGS.log_dir + 'train', graph=graph)
        test_summary_writer = tf.summary.FileWriter(
            FLAGS.log_dir + 'test', graph=graph)

        init = tf.global_variables_initializer()
        # saver
        saver = tf.train.Saver()

        # Session
        with tf.Session() as sess:
            sess.run(init)
            for epoch_steps in range(FLAGS.total_epoches):
                # # shuffle
                np.random.shuffle(concat)

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
                mape_loss_sum = 0.0
                test_batches_amount = len(test_raw_data) // FLAGS.batch_size
                for i in range(test_batches_amount):
                    current_X_batch = test_raw_data[i:i + FLAGS.batch_size]
                    current_Y_batch = test_label_data[i:i + FLAGS.batch_size]
                    test_loss_value, mape_loss_value = sess.run([loss_op, mape_op], feed_dict={
                        X_ph: current_X_batch, Y_ph: current_Y_batch})
                    test_loss_sum += test_loss_value
                    mape_loss_sum += mape_loss_value

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
                mape_mean = (mape_loss_sum / test_batches_amount) * 100.0
                test_scalar_summary = tf.Summary()
                test_scalar_summary.value.add(
                    simple_value=test_mean_loss, tag="mean loss")
                test_scalar_summary.value.add(
                    simple_value=mape_mean, tag="mean mape (%)")
                test_summary_writer.add_summary(
                    test_scalar_summary, global_step=steps)
                test_summary_writer.flush()

                print("ephoches: ", epoch_steps, "trainng loss: ", train_mean_loss,
                      "testing loss: ", test_mean_loss, "testing mape: ", mape_mean, "%")

                if (epoch_steps + 1) % 50 == 0:
                    # Save the variables to disk.
                    save_path = saver.save(
                        sess, FLAGS.checkpoints_dir, global_step=epoch_steps)
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
