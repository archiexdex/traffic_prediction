from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import convlstm

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('log_dir', 'log',
                           "training log directory")
tf.app.flags.DEFINE_integer('batch_size', 128,
                            "mini-batch size")
tf.app.flags.DEFINE_integer('total_epoches', 10000,
                            "")
tf.app.flags.DEFINE_integer('hidden_size', 28,
                            "size of LSTM hidden memory")
tf.app.flags.DEFINE_integer('rnn_layers', 2,
                            "number of stacked lstm")
tf.app.flags.DEFINE_integer('num_steps', 10,
                            "total steps of time")
tf.app.flags.DEFINE_boolean('is_float32', True,
                            "data type of the LSTM state, float32 if true, float16 otherwise")
tf.app.flags.DEFINE_float('learning_rate', 0.1,
                          "learning rate of RMSPropOptimizer")
tf.app.flags.DEFINE_float('decay_rate', 0.99,
                          "decay rate of RMSPropOptimizer")
tf.app.flags.DEFINE_float('momentum', 0.0,
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
        
        ## initialize list
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
                    vec[t+j][index] = tmp 
                    week_list[t+j][index] = (week + int(j / data_per_day)) % 7
                    time[t+j][index] = j % data_per_day
                index = index + 1
            elif ed < i / 2:
                break
            else:
                binaryfile.read(2)



class TestingConfig(object):
    """
    testing config
    """
    batch_size = FLAGS.batch_size
    hidden_size = FLAGS.hidden_size
    is_float32 = FLAGS.is_float32
    num_steps = FLAGS.num_steps
    learning_rate = FLAGS.learning_rate
    momentum = FLAGS.momentum
    decay_rate = FLAGS.decay_rate
    rnn_layers = FLAGS.rnn_layers


def main(_):
    
    raw_data = np.load("raw_data.npy")
    label_data = np.load("label_data.npy")
    label_data = label_data[:,:,1]
    print(label_data.shape[0], label_data.shape[1])

    X = tf.placeholder(dtype=tf.float32, shape=[
        FLAGS.batch_size, FLAGS.num_steps, FLAGS.hidden_size, 5], name='input_data')
    Y = tf.placeholder(dtype=tf.float32, shape=[
        FLAGS.batch_size, FLAGS.hidden_size], name='label_data')

    model = convlstm.TFPModel(TestingConfig(), is_training=True)
    logits_op = model.inference(inputs=X)
    loss_op = model.losses(logits=logits_op, labels=Y)
    train_op = model.train(loss=loss_op, global_step=None)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        for k in range(FLAGS.total_epoches):
            loss_value = None
            i = 0
            while (i+1) * FLAGS.batch_size < len(raw_data):
            # for i in range(1000):
                current_X_batch = np.zeros([128, 10, 28, 5], dtype=np.float)
                current_Y_batch = np.zeros([128, 28], dtype=np.float)
                
                current_X_batch = raw_data[i:i+FLAGS.batch_size]
                current_Y_batch = label_data[i:i+FLAGS.batch_size]

                _, loss_value = sess.run([train_op, loss_op], feed_dict={
                    X: current_X_batch, Y: current_Y_batch})
                # print(i)
                i += 1

            print("iterator : ", k, "train loss : ", loss_value)

    # TODO: https://www.tensorflow.org/api_docs/python/tf/trai/Supervisor
    # sv = Supervisor(logdir=FLAGS.log_dir)
    # with sv.managed_session(FLAGS.master) as sess:
    #     while not sv.should_stop():
    #         sess.run(<my_train_op>)


if __name__ == "__main__":
    tf.app.run()
