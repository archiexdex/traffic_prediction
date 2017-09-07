from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import json
import numpy as np

class Norm(object):
    """
    TODO
    """
    def __init__(self):
        self.key = ["time", "density", "flow", "speed", "week"]
        self.norm = {}
        with open("/home/xdex/Desktop/traffic_flow_detection/taipei/training_data/new_raw_data/vd_base/norm.json", 'r') as fp:
            self.norm = json.load(fp)

    def data_normalization(self, data, file_name):
        # and dump each pair(mean, std) to json for testing
        for i in range(5):
            # temp_mean = np.mean(data[:, :, :, i])
            # temp_std = np.std(data[:, :, :, i])
            temp_mean = self.norm[file_name][self.key[i]][0]
            temp_std = self.norm[file_name][self.key[i]][1]
            data[:, :, :, i] = (data[:, :, :, i] - temp_mean) / temp_std
            print(i, temp_mean, temp_std)
            self.norm[file_name][self.key[i]] = [temp_mean, temp_std]
        return data


    def data_recover(self, data, file_name):
        # and dump each pair(mean, std) to json for testing
        for i in range(5):
            # temp_mean = np.mean(data[:, :, :, i])
            # temp_std = np.std(data[:, :, :, i])
            temp_mean = self.norm[file_name][self.key[i]][0]
            temp_std = self.norm[file_name][self.key[i]][1]
            data[:, :, :, i] = (data[:, :, :, i] - temp_mean) / temp_std
            print(i, temp_mean, temp_std)
            self.norm[file_name][self.key[i]] = [temp_mean, temp_std]
        return data
