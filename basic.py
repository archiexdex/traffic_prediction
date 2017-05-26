from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('log_dir', 'log',
                           "training log directory")
tf.app.flags.DEFINE_integer('batch_size', 128,
                            "mini-batch size")
tf.app.flags.DEFINE_integer('total_epoches', 10000,
                            "")
tf.app.flags.DEFINE_integer('hidden_size', 61,
                            "size of LSTM hidden memory")
tf.app.flags.DEFINE_integer('num_steps', 10,
                            "total steps of time")
tf.app.flags.DEFINE_integer('predict_time', 5,
                            "the predict time")
tf.app.flags.DEFINE_boolean('is_float32', True,
                            "data type of the LSTM state, float32 if true, float16 otherwise")
tf.app.flags.DEFINE_float('learning_rate', 0.1,
                          "learning rate of RMSPropOptimizer")
tf.app.flags.DEFINE_float('decay_rate', 0.99,
                          "decay rate of RMSPropOptimizer")
tf.app.flags.DEFINE_float('momentum', 0.0,
                          "momentum of RMSPropOptimizer")


def read_file(filename, vec, week_list, time_list, week, st, ed):
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
        vt = len(vec)
        wt = len(week_list)
        tt = len(time_list)
        for i in range(day_max):
            vec.append([0] * dis)
            week_list.append([0] * dis)
            time_list.append([0] * dis)
                        
        index = 0
        for i in range(VD_size):
            
            if st <= i / 2 and i / 2 <= ed:
                for j in range(day_max):
                    ptr = binaryfile.read(2)
                    tmp = int.from_bytes(ptr, byteorder='little')
                    vec[vt+j][index] = tmp 
                    week_list[wt+j][index] = (week + int(j / data_per_day)) % 7
                    time_list[tt+j][index] = j % data_per_day
                index = index + 1
            elif ed < i / 2:
                break
            else:
                binaryfile.read(2)


class TFPModel(object):
    """
    The Traffic Flow Prediction Modle
    """

    def __init__(self, config, is_training=True):
        """
        Param:
            config:
        """
        self.is_training = is_training
        self.batch_size = config.batch_size
        self.hidden_size = config.hidden_size
        self.num_steps = config.num_steps
        if config.is_float32:
            self.data_type = tf.float32
        else:
            self.data_type = tf.float16
        self.learning_rate = config.learning_rate
        self.decay_rate = config.decay_rate
        self.momentum = config.momentum

    def lstm_cell(self):
        return tf.contrib.rnn.BasicLSTMCell(self.hidden_size, forget_bias=0.5, state_is_tuple=True,
                                            reuse=tf.get_variable_scope().reuse)

    def predict(self, inputs):
        """
        Param:
            inputs: [batch_size, time_step, milage] = [128, 15, 61]
        """
        logits_list = []
        cell = self.lstm_cell()

        state = cell.zero_state(FLAGS.batch_size, self.data_type)  # [128, 61]
        with tf.variable_scope('LSTM') as scope:
            for time_step in range(self.num_steps):
                if time_step > 0 or not self.is_training:
                    tf.get_variable_scope().reuse_variables()
                cell_out, state = cell(
                    inputs[:, time_step, :], state, scope=scope)
                logits_list.append(cell_out)

        logits_list = tf.transpose(logits_list, perm=[1, 0, 2])
        return logits_list

    def loss_fn(self, logits, labels):
        """
        Param:
            logits:
            labels:
        """
        losses = tf.squared_difference(logits, labels)
        mean_loss_op = tf.reduce_mean(losses)
        return mean_loss_op

    def train(self, loss, global_step=None):
        """
        Param:
            loss:
        """
        train_op = tf.train.RMSPropOptimizer(
            self.learning_rate, self.decay_rate, self.momentum, 1e-10).minimize(loss, global_step=global_step)
        return train_op


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


def main(_):

    # TODO: read data into raw_data
    # raw_data as [time, milage, density]

    # Initialize lists
    density_list = []
    flow_list = []
    speed_list = []
    week_list = []
    time_list = []

    # Read files
    read_file("density_N5_N_2012_1_12.bin", density_list, [], [], 0, 15, 28.5)
    read_file("flow_N5_N_2012_1_12.bin"   , flow_list, [], [], 0, 15, 28.5)
    read_file("speed_N5_N_2012_1_12.bin", speed_list, week_list, time_list, 0, 15, 28.5)

    read_file("density_N5_N_2013_1_12.bin", density_list, [], [], 2, 15, 28.5)
    read_file("flow_N5_N_2013_1_12.bin"   , flow_list, [], [], 2, 15, 28.5)
    read_file("speed_N5_N_2013_1_12.bin", speed_list, week_list, time_list, 2, 15, 28.5)

    read_file("density_N5_N_2014_1_12.bin", density_list, [], [], 3, 15, 28.5)
    read_file("flow_N5_N_2014_1_12.bin"   , flow_list, [], [], 3, 15, 28.5)
    read_file("speed_N5_N_2014_1_12.bin", speed_list, week_list, time_list, 3, 15, 28.5)
    
    # fix data
    # data[i][10] are always 0 and data[i][13] in 2012 are always 0
    for i in range(len(speed_list)):
        density_list[i][10] = int((density_list[i][9] + density_list[i][11]) / 2) if density_list[i][10] is 0 else density_list[i][10]
        density_list[i][13] = int((density_list[i][12] + density_list[i][14]) / 2) if density_list[i][13] is 0 else density_list[i][13]
        flow_list[i][10] = int((flow_list[i][9] + flow_list[i][11]) / 2) if flow_list[i][10] is 0 else flow_list[i][10]
        flow_list[i][13] = int((flow_list[i][12] + flow_list[i][14]) / 2) if flow_list[i][13] is 0 else flow_list[i][13]
        speed_list[i][10] = int((speed_list[i][9] + speed_list[i][11]) / 2) if speed_list[i][10] is 0 else speed_list[i][10]
        speed_list[i][13] = int((speed_list[i][12] + speed_list[i][14]) / 2) if speed_list[i][13] is 0 else speed_list[i][13]

    # merge different dimention data in one
    raw_data = np.stack((density_list, flow_list, speed_list, week_list, time_list), axis=2)
    
    print("raw_data ", len(raw_data))

    # distribute data to each batch and label
    batch_data = []
    label_data = []
    for i in range(len(raw_data) - FLAGS.num_steps - FLAGS.predict_time):
        batch_data.append(raw_data[i:i+FLAGS.num_steps])
        label_data.append(raw_data[i+FLAGS.num_steps+FLAGS.predict_time])

    print("batch_data size ", len(batch_data) )

    # delete illegal batch and coresponding label
    x = np.array(batch_data)
    y = np.array(label_data)
    c = 0
    p = []
    for i in x:
        flg = False
        for j in i:
            for k in j:
                # density, flow, speed, week, time
                t = np.argwhere( (k[0] is 0 or 100 < k[0]) or (k[0] is 0 or 40 * 2 < k[1]) or (k[2] is 0 or 120 < k[2]) )
                if len(t) > 0:
                    flg = True
                    break
            if flg:
                break
        if flg:
            p.append(c)    
        c += 1
        print(c)
    xx = np.delete(x, p, 0)
    yy = np.delete(y, p, 0)
    
    np.save("raw_data", xx)
    np.save("label_data", yy)

    input("@@>>>")
    speed_list = np.array(speed_list).astype(np.float)
    
    train_data, valid_data, test_data = np.split(
        speed_list, [int(speed_list.shape[0] * 8 / 10), int(speed_list.shape[0] * 9 / 10)])

    print (train_data.shape, valid_data.shape, test_data.shape)
    # TODO: maybe train_data.shape[0] isn't the multiple of FLAGS.num_steps
    input_amount = int((train_data.shape[0] - 2) / FLAGS.num_steps)
    batches_in_epoch = int(input_amount / FLAGS.batch_size) - 1

    train_data = train_data[:-2]
    total_inputs = np.reshape(train_data,
                              [input_amount,
                               FLAGS.num_steps,
                               (train_data.shape[1])])
    total_inputs_X = total_inputs[:batches_in_epoch * 128]
    total_inputs_Y = total_inputs[1:
                                  batches_in_epoch * 128 + 1]
    total_inputs_concat = np.concatenate(
        (total_inputs_X, total_inputs_Y), axis=1)
    np.random.shuffle(total_inputs_concat)

    total_inputs_split = np.hsplit(total_inputs_concat, 2)
    total_inputs_X = total_inputs_split[0]
    total_inputs_Y = total_inputs_split[1]

    model_config = TestingConfig()

    # TODO: data preprocessing X and T
    X = tf.placeholder(dtype=tf.float32, shape=[
                       None, FLAGS.num_steps, FLAGS.hidden_size], name='DFS')
    Y = tf.placeholder(dtype=tf.float32, shape=[
                       None, FLAGS.num_steps, FLAGS.hidden_size], name='DFS')
    valid_X = tf.placeholder(dtype=tf.float32, shape=[
        None, FLAGS.num_steps, FLAGS.hidden_size], name='DFS')
    valid_Y = tf.placeholder(dtype=tf.float32, shape=[
        None, FLAGS.num_steps, FLAGS.hidden_size], name='DFS')

    model = TFPModel(model_config, is_training=True)
    logits = model.predict(X)
    losses = model.loss_fn(logits, Y)

    valid_model = TFPModel(model_config, is_training=False)
    valid_logits = valid_model.predict(valid_X)
    valid_losses = valid_model.loss_fn(valid_logits, valid_Y)
    train_op = model.train(losses)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        for k in range(FLAGS.total_epoches):
            loss_value = None
            for i in range(batches_in_epoch):
                start_idx = i * FLAGS.batch_size
                end_idx = (i + 1) * FLAGS.batch_size

                current_X_batch = total_inputs_X[start_idx:end_idx]
                current_Y_batch = total_inputs_Y[start_idx:end_idx]

                _, loss_value = sess.run([train_op, losses], feed_dict={
                                         X: current_X_batch, Y: current_Y_batch})

            valid_amount = int((valid_data.shape[0] - 4) / FLAGS.num_steps)
            valid_data = valid_data[:-4]
            valid_data = np.reshape(valid_data,
                                    [valid_amount,
                                     FLAGS.num_steps,
                                     (valid_data.shape[1])])
            X_valid_data = valid_data[:128]
            Y_valid_data = valid_data[1:129]

            valid_loss_value = sess.run(losses, feed_dict={
                X: X_valid_data, Y: Y_valid_data})
            print("iterator : ", k, "train loss : ", loss_value, "valid loss : ", valid_loss_value)

    # TODO: https://www.tensorflow.org/api_docs/python/tf/trai/Supervisor
    # sv = Supervisor(logdir=FLAGS.log_dir)
    # with sv.managed_session(FLAGS.master) as sess:
    #     while not sv.should_stop():
    #         sess.run(<my_train_op>)


if __name__ == "__main__":
    tf.app.run()
