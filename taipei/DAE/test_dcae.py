from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import datetime
import time
import json
import numpy as np
import tensorflow as tf
import model_dcae
import utils
import plotly
plotly.tools.set_credentials_file(
    username="XDEX", api_key="GTWbfZgCXH6MQ1VOb9FO")

import plotly.plotly as py
import plotly.graph_objs as go

DRAW_ONLINE_FLAG = False
FOLDER_PATH = 'test/'

FLAGS = tf.app.flags.FLAGS

# path parameters
tf.app.flags.DEFINE_string("train_data", "train_data.npy",
                           "training data name")
tf.app.flags.DEFINE_string("test_data", "test_data.npy",
                           "validation data name")
tf.app.flags.DEFINE_string("train_label", "train_label.npy",
                           "training label data name")
tf.app.flags.DEFINE_string("test_label", "test_label.npy",
                           "testing label data name")
tf.app.flags.DEFINE_string('data_dir', '/home/xdex/Desktop/traffic_flow_detection/taipei/training_data/new_raw_data/vd_base/',
                           "data directory")
tf.app.flags.DEFINE_string('checkpoints_dir', 'v5/checkpoints/',
                           "training checkpoints directory")
tf.app.flags.DEFINE_string('log_dir', 'v5/log/',
                           "summary directory")
tf.app.flags.DEFINE_string('restore_path', 'v1/checkpoints/model.ckpt-12800',
                           "path of saving model eg: checkpoints/model.ckpt-5")
# data augmentation and corruption
tf.app.flags.DEFINE_integer('aug_ratio', 1,
                            "the ratio of data augmentation")
tf.app.flags.DEFINE_integer('corrupt_amount', 100,
                            "the amount of corrupted data")
# training parameters
FILTER_NUMBERS = [32, 64, 128]
FILTER_STRIDES = [1, 2, 2]
tf.app.flags.DEFINE_integer('batch_size', 512,
                            "mini-batch size")
tf.app.flags.DEFINE_integer('total_epoches', 100,
                            "total training epoches")
tf.app.flags.DEFINE_integer('save_freq', 25,
                            "number of epoches to saving model")
tf.app.flags.DEFINE_float('learning_rate', 0.001,
                          "learning rate of AdamOptimizer")
tf.app.flags.DEFINE_bool('if_norm_label', True,
                          "if normalize label data")
# tf.app.flags.DEFINE_integer('num_gpus', 2,
#                             "multi gpu")

def generate_input_and_label(all_data, aug_ratio, corrupt_amount, policy='random_vd'):
    print('all_data.shape:', all_data.shape)
    # corrupt_list
    corrupt_list = []
    # data augmentation
    aug_data = []
    for one_data in all_data:
        aug_data.append([one_data for _ in range(aug_ratio)])
    aug_data = np.concatenate(aug_data, axis=0)
    raw_data = np.array(aug_data)
    print('raw_data.shape:', raw_data.shape)
    if policy == 'random_data':
        # randomly corrupt target data
        for one_data in aug_data:
            corrupt_target = np.random.randint(all_data.shape[1] * all_data.shape[2],
                                               size=corrupt_amount)
            corrupt_target = np.stack(
                [corrupt_target // all_data.shape[2], corrupt_target % all_data.shape[2]], axis=1)
            # corrupt target as [0, 0, 0, time, weekday]
            for target in corrupt_target:
                one_data[target[0], target[1], 1:4] = 0.0
            # save corrupt target
            corrupt_list.append(corrupt_target)
        corrupt_data = aug_data
    elif policy == 'random_vd':
        # randomly corrupt 5 target vd
        for one_data in aug_data:
            corrupt_target = np.random.randint(all_data.shape[1], size=corrupt_amount//12)
            # corrupt target as [0, 0, 0, time, weekday]
            one_data[corrupt_target, :, 1:4] = 0.0
            # save corrupt target
            corrupt_list.append(corrupt_target)
        corrupt_data = aug_data

    return corrupt_data, raw_data, corrupt_list

class TrainingConfig(object):
    """
    Training config
    """

    def __init__(self, filter_numbers, filter_strides, input_shape):
        self.filter_numbers = filter_numbers
        self.filter_strides = filter_strides
        self.input_shape = input_shape
        self.data_dir = FLAGS.data_dir
        self.checkpoints_dir = FLAGS.checkpoints_dir
        self.log_dir = FLAGS.log_dir
        self.restore_path = FLAGS.restore_path
        self.aug_ratio = FLAGS.aug_ratio
        self.corrupt_amount = FLAGS.corrupt_amount
        self.batch_size = FLAGS.batch_size
        self.total_epoches = FLAGS.total_epoches
        self.save_freq = FLAGS.save_freq
        self.learning_rate = FLAGS.learning_rate
        self.if_norm_label = FLAGS.if_norm_label

    def show(self):
        print("filter_numbers:", self.filter_numbers)
        print("filter_strides:", self.filter_strides)
        print("input_shape:", self.input_shape)
        print("data_dir:", self.data_dir)
        print("checkpoints_dir:", self.checkpoints_dir)
        print("log_dir:", self.log_dir)
        print("restore_path:", self.restore_path)
        print("aug_ratio:", self.aug_ratio)
        print("corrupt_amount:", self.corrupt_amount)
        print("batch_size:", self.batch_size)
        print("total_epoches:", self.total_epoches)
        print("save_freq:", self.save_freq)
        print("learning_rate:", self.learning_rate)
        print("if_norm_label:", self.if_norm_label)


def plot_result_cmp_label(results, labels, corrupt_list):
    """
    """
    START_TIME = time.mktime( datetime.datetime.strptime("2015-12-01 00:00:00", "%Y-%m-%d %H:%M:%S").timetuple() )
    i = 50
    vd = 67
    # TODO: random_vd or random_data
    target_corrupt_list = []
    for k in corrupt_list[i]:
        
    while i < i+101:
        # Add data
        target_result_flow = results[i, vd, :, 2]
        target_time_list = []
        for k in labels[i, vd, :, 0]:
            target_time_list.append(datetime.datetime.fromtimestamp(k * 300 + START_TIME).strftime("%Y-%m-%d %H:%M:%S"))
        # target_time_list   = datetime.datetime.fromtimestamp(labels[i, vd, :, 0] * 300 + START_TIME).strftime("%Y-%m-%d %H:%M:%S")
        target_label_flow  = labels[i, vd, :, 3]


        # Create and style traces
        trace_flow = go.Scatter(
            x=target_time_list,
            y=target_result_flow,
            name='Flow',
            line=dict(
                color=('rgb(255, 0, 0)'),
                width=3)
        )
        
        trace_label = go.Scatter(
            x=target_time_list,
            y=target_label_flow,
            name='Label',
            line=dict(
                color=('rgb(0, 255, 0)'),
                width=3)
        )
        data = [trace_flow, trace_label]

        
        # Edit the layout
        layout = dict(title="VD_ID_%d_DATA_ID_%d.html" % (vd, i),
                      xaxis=dict(title='Time'),
                      yaxis=dict(title='Value'),
                      )

        fig = dict(data=data, layout=layout)
        if DRAW_ONLINE_FLAG:
            py.plot(fig, filename=FOLDER_PATH + "VD_ID_%d_DATA_ID_%d.html" % (vd, i))
        else:
            plotly.offline.plot(
                fig, filename=FOLDER_PATH + "VD_ID_%d_DATA_ID_%d.html" % (vd, i))

        i += 1
        input("!")

def main(_):
    with tf.get_default_graph().as_default() as graph:
        # load data
        train_data = np.load(FLAGS.data_dir + FLAGS.train_data)
        # generate many pollute data and pure data
        polluted_train_input, pure_train_input, corrupt_list = generate_input_and_label(
            train_data, FLAGS.aug_ratio, FLAGS.corrupt_amount)
        # data normalization
        Norm_er = utils.Norm()
        polluted_train_input = Norm_er.data_normalization(polluted_train_input)
        if FLAGS.if_norm_label:
            pure_train_input = Norm_er.data_normalization(pure_train_input)[:, :, :, 1:4]
        else:
            pure_train_input = pure_train_input[:, :, :, 1:4]

        # number of batches
        train_num_batch = polluted_train_input.shape[0] // FLAGS.batch_size
        print("train_batch:", train_num_batch)
        # config setting
        config = TrainingConfig(
            FILTER_NUMBERS, FILTER_STRIDES, polluted_train_input.shape)
        config.show()
        # model
        model = model_dcae.DCAEModel(config, graph=graph)
        # Add an op to initialize the variables.
        init = tf.global_variables_initializer()
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

        # Session
        with tf.Session() as sess:
            sess.run(init)
            # restore model if exist
            if FLAGS.restore_path is not None:
                saver.restore(sess, FLAGS.restore_path)
                print("Model restored:", FLAGS.restore_path)
            # prediction on test dataset
            result_all = []
            result_loss = []
            result_loss_sum = 0
            each_loss_sum = np.zeros(shape=[3], dtype=np.float)
            for b in range(train_num_batch):
                batch_idx = b * FLAGS.batch_size
                train_data_batch = polluted_train_input[batch_idx:batch_idx + FLAGS.batch_size]
                label_data_batch = pure_train_input[batch_idx:batch_idx +
                                                    FLAGS.batch_size]
                # train one batch
                result = model.predict(sess, train_data_batch)
                result_all.append(result)

                # compute loss
                loss, each_loss = model.compute_loss(
                    sess, train_data_batch, label_data_batch)
                if FLAGS.if_norm_label:
                    each_loss = Norm_er.data_recover(each_loss)
                
                result_loss.append(loss)
                result_loss_sum += loss
                each_loss_sum += each_loss

            result_all = np.array(result_all)
            result_loss = np.array(result_loss)

            print("test mean loss: %f" % (result_loss_sum / train_num_batch))
            print("%f density_loss, %f flow_loss, %f speed_loss" %
                      (each_loss_sum[0] / train_num_batch,
                       each_loss_sum[1] / train_num_batch,
                       each_loss_sum[2] / train_num_batch))
            result_all = np.reshape(
                result_all, (result_all.shape[0] * result_all.shape[1], result_all.shape[2], result_all.shape[3], result_all.shape[4]))
            print(result_all.shape)
            print(train_data.shape)
            plot_result_cmp_label(result_all, train_data, corrupt_list)


if __name__ == "__main__":
    if not os.path.exists(FLAGS.checkpoints_dir):
        os.makedirs(FLAGS.checkpoints_dir)
    tf.app.run()
