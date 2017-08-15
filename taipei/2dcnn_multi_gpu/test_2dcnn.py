from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import tensorflow as tf
import datetime
import model_2dcnn
import plotly
plotly.__version__
plotly.tools.set_credentials_file(
    username='ChenChiehYu', api_key='xh9rsxFXY6DNF1qAfUyQ')
import plotly.plotly as py
import plotly.graph_objs as go

DRAW_ONLINE_FLAG = False
FOLDER_PATH = 'test/'

FLAGS = tf.app.flags.FLAGS

# path parameters
tf.app.flags.DEFINE_string("train_data", "train_data.npy",
                           "training data name")
tf.app.flags.DEFINE_string("test_data", "test_data.npy",
                           "testing data name")
tf.app.flags.DEFINE_string("train_label", "train_label.npy",
                           "training label data name")
tf.app.flags.DEFINE_string("test_label", "test_label.npy",
                           "testing label data name")
tf.app.flags.DEFINE_string('data_dir', '/home/xdex/Desktop/traffic_flow_detection/taipei/training_data/',
                           "data directory")
tf.app.flags.DEFINE_string('checkpoints_dir', '',
                           "training checkpoints directory")
tf.app.flags.DEFINE_string('log_dir', '',
                           "summary directory")
# training parameters
tf.app.flags.DEFINE_integer('batch_size', 512,
                            "mini-batch size")
tf.app.flags.DEFINE_integer('total_epoches', 0,
                            "total training epoches")
tf.app.flags.DEFINE_integer('save_freq', 0,
                            "number of epoches to saving model")
tf.app.flags.DEFINE_integer('total_interval', 0,
                            "total steps of time")
tf.app.flags.DEFINE_float('learning_rate', 0,
                          "learning rate of AdamOptimizer")
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            "multi gpu")
tf.app.flags.DEFINE_string('restore_path', None,
                           "path of saving model eg: checkpoints/model.ckpt-5")


class ModelConfig(object):
    """
    testing config
    """

    def __init__(self):
        self.data_dir = FLAGS.data_dir
        self.checkpoints_dir = FLAGS.checkpoints_dir
        self.log_dir = FLAGS.log_dir
        self.batch_size = FLAGS.batch_size
        self.total_epoches = FLAGS.total_epoches
        self.save_freq = FLAGS.save_freq
        self.total_interval = FLAGS.total_interval
        self.learning_rate = FLAGS.learning_rate
        self.num_gpus = FLAGS.num_gpus

    def show(self):
        print("data_dir:", self.data_dir)
        print("checkpoints_dir:", self.checkpoints_dir)
        print("log_dir:", self.log_dir)
        print("batch_size:", self.batch_size)
        print("total_epoches:", self.total_epoches)
        print("save_freq:", self.save_freq)
        print("total_interval:", self.total_interval)
        print("learning_rate:", self.learning_rate)
        print("num_gpus:", self.num_gpus)

def plot_result_cmp_label(results, labels):
    """
    """
    for i in range(results.shape[1]): # for each target vd
        # Add data
        target_vd_result = results[:, i]
        target_vd_label = labels[:, i]
        # time_list = []
        # for _, v in enumerate(target_vd_timestamp):
        #     time_list.append(datetime.datetime.fromtimestamp(
        #         v).strftime("%Y-%m-%d %H:%M:%S"))

        # Create and style traces
        trace_flow = go.Scatter(
            # x=time_list,
            y=target_vd_result,
            name='Flow',
            line=dict(
                color=('rgb(24, 12, 205)'),
                width=3)
        )
        trace_label= go.Scatter(
            # x=time_list,
            y=target_vd_label,
            name='Label',
            line=dict(
                color=('rgb(24, 205, 12)'),
                width=3)
        )
        data = [trace_flow, trace_label]

        # Edit the layout
        layout = dict(title="VD_ID: %d" % i,
                    xaxis=dict(title='Time'),
                    yaxis=dict(title='Value'),
                    )

        fig = dict(data=data, layout=layout)
        if DRAW_ONLINE_FLAG:
            py.plot(fig, filename=FOLDER_PATH + "VD_ID: %d.html" % i)
        else:
            plotly.offline.plot(fig, filename=FOLDER_PATH + "VD_ID: %d.html" % i)


def main(_):
    with tf.get_default_graph().as_default() as graph:
        # config setting
        config = ModelConfig()
        config.show()
        # load data
        test_data = np.load(FLAGS.data_dir + FLAGS.test_data)[:, :, :, :]
        test_label = np.load(FLAGS.data_dir + FLAGS.test_label)
        # number of batches
        test_num_batch = test_data.shape[0] // FLAGS.batch_size
        print('test_num_batch:', test_num_batch)
        # model
        model = model_2dcnn.TFPModel(config, graph=graph)
        # Add an op to initialize the variables.
        init = tf.global_variables_initializer()

        # Session
        with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
            sess.run(init)
            # restore model if exist
            if FLAGS.restore_path is not None:
                saver.restore(sess, FLAGS.restore_path)
                print("Model restored:", FLAGS.restore_path)
            # prediction on test dataset
            result_all = []
            for test_b in range(test_num_batch):
                batch_idx = test_b * FLAGS.batch_size
                # input, label
                test_data_batch = test_data[batch_idx:batch_idx +
                                            FLAGS.batch_size]
                result = model.predict(
                    sess, test_data_batch)
                result_all.append(result)
            result_all = np.array(result_all)
            result_all = np.concatenate(result_all, axis=0)
            # evaluate on test dataset
            test_each_vd_losses_sum = []
            test_loss_sum = 0.0
            for test_b in range(test_num_batch):
                batch_idx = test_b * FLAGS.batch_size
                # input, label
                test_data_batch = test_data[batch_idx:batch_idx +
                                            FLAGS.batch_size]
                test_label_batch = test_label[batch_idx:batch_idx +
                                              FLAGS.batch_size]
                test_each_vd_losses, test_losses = model.compute_loss(
                    sess, test_data_batch, test_label_batch)
                test_each_vd_losses_sum.append(test_each_vd_losses)
                test_loss_sum += test_losses
            test_each_vd_losses_sum = np.array(test_each_vd_losses_sum)
            test_each_vd_losses_mean = np.mean(test_each_vd_losses_sum, axis=0)
            # log
            print("test mean loss: %f" % (test_loss_sum / test_num_batch))
            print("each test vd's mean loss:")
            print(test_each_vd_losses_mean)
            
            plot_result_cmp_label(result_all, test_label)


if __name__ == "__main__":
    if not os.path.exists(FOLDER_PATH):
        os.mkdir(FOLDER_PATH)
    tf.app.run()
