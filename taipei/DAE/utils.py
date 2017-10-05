from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import json
import numpy as np
import tensorflow as tf


class Norm(object):
    """
    TODO
    """

    def __init__(self):
        self.key = ["time", "density", "flow", "speed", "week"]
        self.norm = {}
        with open("/home/xdex/Desktop/traffic_flow_detection/taipei/training_data/new_raw_data/vd_base/norm.json", 'r') as fp:
            self.norm = json.load(fp)

    def data_normalization(self, data):
        for i in range(5):
            temp_mean = self.norm['train'][self.key[i]][0]
            temp_std = self.norm['train'][self.key[i]][1]
            data[:, :, :, i] = (data[:, :, :, i] - temp_mean) / temp_std
            print(i, temp_mean, temp_std)
        return data

    def data_recover(self, data):
        for i in range(1, 4):
            temp_mean = self.norm['train'][self.key[i]][0]
            temp_std = self.norm['train'][self.key[i]][1]
            data[i-1] = data[i-1] * temp_std + temp_mean
        return data

    def logits_recover(self, logits):
        # logits = [batch, vds, times, 3]
        mean_list = []
        std_list= []
        for i in range(1, 4):
            mean_list.append(self.norm['train'][self.key[i]][0])
            std_list.append(self.norm['train'][self.key[i]][1])
        logits = tf.multiply(logits, std_list) + mean_list
        return logits



def generate_input_and_label(all_data, aug_ratio, corrupt_ratio, policy='random_vd'):
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
                                               size= int( all_data.shape[1] * all_data.shape[2] * corrupt_ratio ) )
            corrupt_tmp = []
            corrupt_target = np.stack(
                [corrupt_target // all_data.shape[2], corrupt_target % (all_data.shape[2]-2)], axis=1)
            # corrupt target as [time, 0, 0, 0, weekday, missing=True]
            for target in corrupt_target:
                one_data[target[0], target[1]+1, 1:4] = 0.0
                one_data[target[0], target[1]+1, 5] = 1
                corrupt_tmp.append([target[0], target[1], target[1]+1])
            corrupt_list.append(corrupt_tmp)
        corrupt_data = aug_data
    elif policy == 'random_vd':
        # randomly corrupt 5 target vd
        for one_data in aug_data:
            corrupt_tmp = []
            corrupt_target = np.random.randint(all_data.shape[1], size=int( all_data.shape[1] * corrupt_ratio ) ) 
            # corrupt target as [0, 0, 0, time, weekday]
            for target in corrupt_target:
                # random start time and end time
                corrupt_target_range = np.random.randint(6, size=1)
                one_data[target, corrupt_target_range[0]+1:11, 1:4] = 0.0
                one_data[target, corrupt_target_range[0]+1:11, 5] = 1
                corrupt_tmp.append([target, corrupt_target_range[0], 11])
            # save corrupt target
            corrupt_list.append(corrupt_tmp)
        corrupt_data = aug_data

    return corrupt_data, raw_data, corrupt_list