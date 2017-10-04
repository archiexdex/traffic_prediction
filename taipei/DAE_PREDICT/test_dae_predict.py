from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import shutil
import numpy as np
import tensorflow as tf
import model_dae_predict

FLAGS = tf.app.flags.FLAGS

# path parameters
tf.app.flags.DEFINE_string('data_dir', '/home/xdex/Desktop/traffic_flow_detection/taipei/training_data/old_Taipei_data/vd_base/',
                           "data directory")
tf.app.flags.DEFINE_string("train_data", "train_data_train_0_label_100.npy",
                           "training data name")
tf.app.flags.DEFINE_string("test_data", "test_data_train_0_label_100.npy",
                           "testing data name")
tf.app.flags.DEFINE_string("train_label", "train_label_train_0_label_100.npy",
                           "training label data name")
tf.app.flags.DEFINE_string("test_label", "test_label_train_0_label_100.npy",
                           "testing label data name")
tf.app.flags.DEFINE_string('vis_dir', 'v3/vis/',
                           "visualization directory")
# training parameters
tf.app.flags.DEFINE_string('restore_path', None,
                           "path of saved model (DAE+PREDICT) eg: 'v3/checkpoints/model.ckpt-11500'")
tf.app.flags.DEFINE_integer('batch_size', 512,
                            "mini-batch size")

def vis_result(labels, predictions):
    """ vis TODO
    params
    ------
        labels : float, shape=(#, vds=18, intervals=4)
        predictions : float, shape=(#, vds=18, intervals=4)
    """
    pass

def main(_):
    with tf.get_default_graph().as_default() as graph:
        # load data
        # train_data = np.load(FLAGS.data_dir + FLAGS.train_data)[:, :, :, :6]
        # train_label = np.load(FLAGS.data_dir + FLAGS.train_label)[:, :, :, 2]
        test_data = np.load(FLAGS.data_dir + FLAGS.test_data)[:, :, :, :6]
        test_label = np.load(FLAGS.data_dir + FLAGS.test_label)[:, :, :, 2]
        # number of batches
        # train_num_batch = train_data.shape[0] // FLAGS.batch_size
        # print(train_num_batch)
        test_num_batch = test_data.shape[0] // FLAGS.batch_size
        print(test_num_batch)
        # # config setting
        # model
        tf.train.import_meta_graph(FLAGS.restore_path + '.meta')
        X_ph = graph.get_tensor_by_name('corrupt_data:0')
        Y_ph = graph.get_tensor_by_name('PREDICT/label_data:0')
        result = graph.get_tensor_by_name('PREDICT/reshape/Reshape:0')
        each_vd_loss = graph.get_tensor_by_name('PREDICT/l2_loss/Mean:0')
        l2_loss = graph.get_tensor_by_name('PREDICT/l2_loss/Mean_1:0')

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

        # Session
        with tf.Session() as sess:
            # restore model if exist
            if FLAGS.restore_path is not None:
                saver.restore(sess, FLAGS.restore_path)
                print("Model restored:", FLAGS.restore_path)

            # time cost evaluation
            start_time = time.time()

            # testing
            all_predictions = []
            test_each_vd_losses_sum = []
            test_loss_sum = 0.0
            for test_b in range(test_num_batch):
                batch_idx = test_b * FLAGS.batch_size
                # input, label
                test_data_batch = test_data[batch_idx:batch_idx +
                                            FLAGS.batch_size]
                test_label_batch = test_label[batch_idx:batch_idx +
                                              FLAGS.batch_size]
                feed_dict = {
                    X_ph: test_data_batch,
                    Y_ph: test_label_batch
                }
                predictions, test_each_vd_losses, test_losses = sess.run(
                    [result, each_vd_loss, l2_loss], feed_dict=feed_dict)

                all_predictions.append(predictions)
                test_each_vd_losses_sum.append(test_each_vd_losses)
                test_loss_sum += test_losses
            # all result
            all_predictions = np.array(all_predictions)
            all_predictions = np.concatenate(all_predictions, axis=0)
            # vis
            vis_result(test_label, all_predictions)

            test_each_vd_losses_sum = np.array(test_each_vd_losses_sum)
            test_each_vd_losses_mean = np.mean(
                test_each_vd_losses_sum, axis=0)

            end_time = time.time()
            # logging per ephoch
            print("test mean loss: %f, time cost: %f(sec)" %
                  (test_loss_sum / test_num_batch, (end_time - start_time)))
            print("each test vd's mean loss:")
            print(test_each_vd_losses_mean)


if __name__ == "__main__":
    if FLAGS.restore_path is None:
        raise AssertionError("FLAGS.restore_path should not be None!!!")
    if os.path.exists(FLAGS.vis_dir):
        shutil.rmtree(FLAGS.vis_dir)
        print('rm -rf "%s" complete!' % FLAGS.vis_dir)
    tf.app.run()
