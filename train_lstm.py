from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import convlstm as convlstm
# import convlstm_bkp as convlstm

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('checkpoints_dir', 'checkpoints/',
                           "training checkpoints directory")
tf.app.flags.DEFINE_integer('batch_size', 128,
                            "mini-batch size")
tf.app.flags.DEFINE_integer('total_epoches', 100,
                            "")
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
        self.batch_size = FLAGS.batch_size
        self.total_epoches = FLAGS.total_epoches
        self.hidden_size = FLAGS.hidden_size
        self.is_float32 = FLAGS.is_float32
        self.num_steps = FLAGS.num_steps
        self.learning_rate = FLAGS.learning_rate
        self.momentum = FLAGS.momentum
        self.decay_rate = FLAGS.decay_rate
        self.rnn_layers = FLAGS.rnn_layers

    def show(self):
        print ("batch_size:", self.batch_size)
        print ("total_epoches:", self.total_epoches)
        print ("hidden_size:", self.hidden_size)
        print ("is_float32:", self.is_float32)
        print ("num_steps:", self.num_steps)
        print ("learning_rate:", self.learning_rate)
        print ("momentum:", self.momentum)
        print ("decay_rate:", self.decay_rate)
        print ("rnn_layers:", self.rnn_layers)


def main(_):
    with tf.get_default_graph().as_default() as graph:
        global_steps = tf.train.get_or_create_global_step(graph=graph)

        raw_data = np.load("raw_data_6.npy")
        label_data = np.load("label_data_6.npy")
        test_raw_data = np.load("test_raw_data_6.npy")
        test_label_data = np.load("test_label_data_6.npy")
        # TODO
        raw_data = raw_data[:, :, :, 1]
        label_data = label_data[:, :, 1]
        test_raw_data = test_raw_data[:, :, :, 1]
        test_label_data = test_label_data[:, :, 1]

        # TODO
        # raw_data_t = raw_data[:, :, :, 1]
        # label_data_t = label_data[:, :, 1]
        # c = np.c_[raw_data_t.reshape(len(raw_data_t), -1),
        #           label_data_t.reshape(len(label_data_t), -1)]
        # np.random.shuffle(c)
        # raw_data = c[:, :raw_data_t.size //
        #              len(raw_data_t)].reshape(raw_data_t.shape)
        # label_data = c[:, raw_data_t.size //
        #                len(raw_data_t):].reshape(label_data_t.shape)
        print(raw_data.shape)
        print(label_data.shape)

        # TODO
        X = tf.placeholder(dtype=tf.float32, shape=[
            FLAGS.batch_size, FLAGS.num_steps, 28 * 1], name='input_data')
        Y = tf.placeholder(dtype=tf.float32, shape=[
            FLAGS.batch_size, 28], name='label_data')

        config = TestingConfig()
        config.show()
        model = convlstm.TFPModel(config, is_training=True)
        logits_op = model.inference(inputs=X)
        loss_op = model.losses(logits=logits_op, labels=Y)
        train_op = model.train(loss=loss_op, global_step=global_steps)
        merged_op = tf.summary.merge_all()
        train_summary_writer = tf.summary.FileWriter('log/train', graph=graph)
        test_summary_writer = tf.summary.FileWriter('log/test', graph=graph)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)
            # training
            for k in range(FLAGS.total_epoches):
                raw_data_t = raw_data
                label_data_t = label_data
                c = np.c_[raw_data_t.reshape(len(raw_data_t), -1),
                          label_data_t.reshape(len(label_data_t), -1)]
                np.random.shuffle(c)
                raw_data = c[:, :raw_data_t.size //
                             len(raw_data_t)].reshape(raw_data_t.shape)
                label_data = c[:, raw_data_t.size //
                               len(raw_data_t):].reshape(label_data_t.shape)

                train_loss_sum = 0.0
                batches_amount = len(raw_data) // FLAGS.batch_size
                for i in range(batches_amount):
                    current_X_batch = raw_data[i:i + FLAGS.batch_size]
                    current_Y_batch = label_data[i:i + FLAGS.batch_size]
                    summary, _, loss_value, steps = sess.run([merged_op, train_op, loss_op, global_steps], feed_dict={
                        X: current_X_batch, Y: current_Y_batch})
                    train_summary_writer.add_summary(summary, global_step=steps)
                    train_loss_sum += loss_value

                # testing ###################
                test_loss_sum = 0.0
                test_batches_amount = len(test_raw_data) // FLAGS.batch_size
                for i in range(test_batches_amount):
                    current_X_batch = test_raw_data[i:i + FLAGS.batch_size]
                    current_Y_batch = test_label_data[i:i + FLAGS.batch_size]
                    test_loss_value = sess.run(loss_op, feed_dict={
                        X: current_X_batch, Y: current_Y_batch})
                    test_loss_sum += test_loss_value
                ################################

                # mean ephoch loss
                train_mean_loss = train_loss_sum / batches_amount
                test_mean_loss = test_loss_sum / test_batches_amount
                print ("ephoches: ", k, "trainng loss: ", train_mean_loss,
                       "testing loss: ", test_mean_loss)
                train_scalar_summary = tf.Summary()
                train_scalar_summary.value.add(
                    simple_value=train_mean_loss, tag="mean loss")
                test_scalar_summary = tf.Summary()
                test_scalar_summary.value.add(
                    simple_value=test_mean_loss, tag="mean loss")
                train_summary_writer.add_summary(train_scalar_summary, global_step=steps)
                test_summary_writer.add_summary(test_scalar_summary, global_step=steps)
                train_summary_writer.flush()
                test_summary_writer.flush()

            # for k in range(1):
            #     i = 0
            #     while (i + 1) < 100:
            #         current_X_batch = np.ones(
            #             [FLAGS.batch_size, 10, 28], dtype=np.float)
            #         current_Y_batch = np.ones(
            #             [FLAGS.batch_size, 28], dtype=np.float) * 3
            #         _ = sess.run([train_op], feed_dict={
            #             X: current_X_batch, Y: current_Y_batch})
            #         i += 1
            #         if i % 5 == 0:
            #             loss_value = sess.run(loss_op, feed_dict={
            #                 X: current_X_batch, Y: current_Y_batch})
            #             print ("i : ", i, "batch loss : ", loss_value)

            # Save the variables to disk.
            save_path = saver.save(
                sess, FLAGS.checkpoints_dir, global_step=steps)
            print ("Model saved in file: %s" % save_path)

        # TODO: https://www.tensorflow.org/api_docs/python/tf/trai/Supervisor
        # sv = Supervisor(logdir=FLAGS.checkpoints_dir)
        # with sv.managed_session(FLAGS.master) as sess:
        #     while not sv.should_stop():
        #         sess.run(<my_train_op>)


if __name__ == "__main__":
    tf.app.run()
