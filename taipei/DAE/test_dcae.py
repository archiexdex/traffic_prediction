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
tf.app.flags.DEFINE_string('data_dir', '/home/xdex/Desktop/traffic_flow_detection/taipei/training_data/old_Taipei_data/vd_base/',
                           "data directory")
# tf.app.flags.DEFINE_string('data_dir', '/home/xdex/Desktop/traffic_flow_detection/taipei/training_data/new_raw_data/vd_base/',
#                            "data directory")
tf.app.flags.DEFINE_string('checkpoints_dir', 'v1/checkpoints/',
                           "training checkpoints directory")
tf.app.flags.DEFINE_string('log_dir', 'v1/log/',
                           "summary directory")
tf.app.flags.DEFINE_string('restore_path', 'v1/checkpoints/model.ckpt-35700',
                           "path of saving model eg: checkpoints/model.ckpt-5")
# data augmentation and corruption
tf.app.flags.DEFINE_integer('aug_ratio', 1,
                            "the ratio of data augmentation")
tf.app.flags.DEFINE_integer('corrupt_amount', 0.1,
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
tf.app.flags.DEFINE_bool('if_norm_label', False,
                         "if normalize label data")
# tf.app.flags.DEFINE_integer('num_gpus', 2,
#                             "multi gpu")
tf.app.flags.DEFINE_bool('if_label_normed', True,
                         "if label data normalized, we need to recover its scale back before compute loss")
tf.app.flags.DEFINE_bool('if_mask_only', True,
                         "if the loss computed on mask only")


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
        self.if_label_normed = FLAGS.if_label_normed
        self.if_mask_only = FLAGS.if_mask_only

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
        print("if_label_normed:", self.if_label_normed)
        print("if_mask_only:", self.if_mask_only)


def plot_result_cmp_label(results, labels, corrupt_list):
    """
    """
    START_TIME = time.mktime(datetime.datetime.strptime(
        "2015-12-01 00:00:00", "%Y-%m-%d %H:%M:%S").timetuple())
    i = 500
    while i < results.shape[0]:
        # vd = 67
        # TODO: random_vd or random_data
        target_corrupt_list = corrupt_list[i]
        flg = 0
        for idx, item in enumerate(target_corrupt_list):
            vd, st_time, ed_time = item
            # Add data
            target_result_flow = results[i, vd, st_time:ed_time + 1, 2]
            # for ptr in range(len(target_result_flow)):
            #     if ptr < st_time or ed_time > ptr:
            #         target_result_flow[ptr] = -1
            target_label_flow = labels[i, vd, :, 3]
            target_time_list = []
            # if not ("08:00" <= datetime.datetime.fromtimestamp(labels[i, vd, 0, -1]).strftime("%H:%M")
            #         and datetime.datetime.fromtimestamp(labels[i, vd, 0, -1]).strftime("%H:%M") <= "22:00"):
            #     i += 1
            #     flg = 1
            #     break

            for k in labels[i, vd, :, -1]:
                target_time_list.append(datetime.datetime.fromtimestamp(
                    k).strftime("%Y-%m-%d %H:%M:%S"))
                print(k)
            target_result_time = [it for kdx, it in enumerate(
                target_time_list) if st_time <= kdx and kdx <= ed_time]

            # Create and style traces
            trace_flow = go.Scatter(
                x=target_result_time,
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
                py.plot(fig, filename=FOLDER_PATH +
                        "VD_ID_%d_DATA_ID_%d.html" % (vd, i))
            else:
                plotly.offline.plot(
                    fig, filename=FOLDER_PATH + "VD_ID_%d_DATA_ID_%d.html" % (vd, i))

        
        print(i)
        if flg == 0:
            i += 1
            input("!")


def merge_result_with_label(results, train_data, corrupt_list):

    for i in range(results.shape[0]):
        # print(i)
        target_corrupt_list = corrupt_list[i]
        for _, item in enumerate(target_corrupt_list):
            vd, st_time, ed_time = item
            # merge data
            # [time, density, flow, speed, weak, mask, timestamp]
            train_data[i, vd, st_time:ed_time + 1, 1:1 +
                       3] = results[i, vd, st_time:ed_time + 1, 0:0 + 3]

    return train_data


def main(_):
    with tf.get_default_graph().as_default() as graph:
        # load data
        train_data = np.load(FLAGS.data_dir + FLAGS.train_data)
        # generate many pollute data and pure data
        polluted_train_input, pure_train_input, corrupt_list = utils.generate_input_and_label(
            train_data, FLAGS.aug_ratio, FLAGS.corrupt_amount, policy='random_data')
        # data normalization
        Norm_er = utils.Norm()
        polluted_train_input = Norm_er.data_normalization(polluted_train_input)[
            :, :, :, :6]
        if FLAGS.if_norm_label:
            pure_train_input = Norm_er.data_normalization(pure_train_input)[
                :, :, :, 1:4]
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
        # model = model_dcae.DCAEModel(config, graph=graph)
        # Add an op to initialize the variables.
        # init = tf.global_variables_initializer()
        # Add ops to save and restore all the variables.
        # saver = tf.train.Saver()

        # Session
        with tf.Session() as sess:
            
            saver = tf.train.import_meta_graph( FLAGS.restore_path + '.meta')
            saver.restore(sess, FLAGS.restore_path)

            model_fn = tf.get_collection("model")[0]
            loss_fn  = tf.get_collection("loss")[0]
            sep_loss_fn = tf.get_collection("sep_loss")[0]
            
            # prediction on test dataset
            result_all = []
            result_loss = []
            result_loss_sum = 0
            each_loss_sum = np.zeros(shape=[3], dtype=np.float)
            for b in range(train_num_batch):
                batch_idx = b * FLAGS.batch_size
                train_data_batch = polluted_train_input[batch_idx:batch_idx +
                                                        FLAGS.batch_size]
                label_data_batch = pure_train_input[batch_idx:batch_idx +
                                                    FLAGS.batch_size]
                # train one batch
                result = sess.run(model_fn, feed_dict={"corrupt_data:0": train_data_batch})
                result_all.append(result)

                # compute loss
                loss, each_loss = sess.run([loss_fn, sep_loss_fn], 
                                feed_dict={"corrupt_data:0":train_data_batch, "raw_data:0":label_data_batch})
                result_loss.append(loss)
                result_loss_sum += loss
                each_loss_sum += each_loss

                # compute interpolation loss
                inter_each_loss_sum = np.zeros(shape=[3], dtype=np.float)
                
                for i in range(train_data_batch.shape[0]):
                    pollute_list = np.argwhere(train_data_batch[i][:,:,5] == 1)
                    # print(pollute_list)
                    vdno = -1
                    each_loss = np.zeros(shape=[3], dtype=np.float)
                    idx = 0
                    for _ in range(len(pollute_list)):
                        if idx == len(pollute_list):
                            break
                        tmp_list = []
                        vdno, ctime = pollute_list[idx]
                        sttime = ctime-1
                        tmp_list.append(pollute_list[idx])
                        jdx = idx + 1
                        for _ in range(len(pollute_list)):
                            if jdx == len(pollute_list):
                                break
                            if vdno == pollute_list[jdx][0] and ctime+1 == pollute_list[jdx][1]:
                                tmp_list.append(pollute_list[jdx])
                                ctime = pollute_list[jdx][1]
                                jdx += 1
                            else:
                                break
                        denominator = ctime + 1 - sttime
                        base = train_data_batch[i][vdno,sttime, 1:1+3]
                        delta = ( train_data_batch[i][vdno,ctime+1, 1:1+3] - base ) / denominator
                        # print(train_data_batch[i][vdno, :, :])
                        # print(vdno, ctime, sttime, denominator, base, delta, tmp_list )
                        for ptr in tmp_list:
                            each_loss += abs( ( base + delta * (ptr[1] - sttime) * denominator ) - label_data_batch[i][ptr[0],ptr[1], :] )
                        idx = jdx
                    each_loss /= len(pollute_list)
                    inter_each_loss_sum += each_loss

            result_all = np.array(result_all)
            result_loss = np.array(result_loss)

            print("test mean loss: %f" % (result_loss_sum / train_num_batch))
            print("test %f density_loss, %f flow_loss, %f speed_loss" %
                  (each_loss_sum[0] / train_num_batch,
                   each_loss_sum[1] / train_num_batch,
                   each_loss_sum[2] / train_num_batch))
            print("linear %f density_loss, %f flow_loss, %f speed_loss" %
                  (inter_each_loss_sum[0] / train_num_batch,
                   inter_each_loss_sum[1] / train_num_batch,
                   inter_each_loss_sum[2] / train_num_batch))
            result_all = np.reshape(
                result_all, (result_all.shape[0] * result_all.shape[1], result_all.shape[2], result_all.shape[3], result_all.shape[4]))
            print(result_all.shape)
            print(train_data.shape)
            # draw the result
            # plot_result_cmp_label(result_all, train_data, corrupt_list)
            # fix data
            # train_data = merge_result_with_label(
            #     result_all, train_data, corrupt_list)
            # np.save(FLAGS.data_dir + "fix_" + FLAGS.train_data, train_data)
            # print(FLAGS.data_dir + "fix_" + FLAGS.train_data + " saved!!")


if __name__ == "__main__":
    if not os.path.exists(FLAGS.checkpoints_dir):
        os.makedirs(FLAGS.checkpoints_dir)
    tf.app.run()
