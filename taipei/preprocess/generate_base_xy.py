"""
generate training/ testing data
based on x_base order and y_base order, then concatenate them together.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import codecs
import json
import matplotlib.pyplot as plt

DATA_PATH='/home/xdex/Desktop/traffic_flow_detection/taipei/'
    

def main():
    """
    main
    """

    ### load raw data
    raw_filename = DATA_PATH + 'fix_raw_data.json'
    vd_id_filename = DATA_PATH + 'reduce_dimension.json'
    with codecs.open(raw_filename, 'r', 'utf-8') as f:
        raw_data = json.load(f)
    # print(np.array(list(raw_data.values())).shape)

    ### read vd id list
    with codecs.open(vd_id_filename, 'r', 'utf-8') as f:
        vd_list_data = json.load(f)
    x_base_list = vd_list_data['x_base']
    y_base_list = vd_list_data['y_base']
    # print(np.array(x_base_list).shape)
    # print(np.array(y_base_list).shape)

    ### gather data into one tensor according to vd id list
    all_data = []
    for idx in x_base_list:
        all_data.append(raw_data[idx])
    for idx in y_base_list:
        all_data.append(raw_data[idx])
    all_data = np.array(all_data)
    # print(all_data.shape)

    ### generate repeared data from shape=[2*vds, intervals, features] to shape=[num_data, 2*vds, 12, features]
    repeated_data = []
    # iter all interval
    for i in range(all_data.shape[1] - 12):
        repeated_data.append(all_data[:, i:i + 12, :])
    repeated_data = np.array(repeated_data)
    # print(repeated_data.shape)

    ### label data
    repeated_label = []
    for i in range(12, all_data.shape[1]):  # iter all interval
        repeated_label.append(all_data[:35, i, 1])  # 1-> flow only
    repeated_label = np.array(repeated_label)
    # print(repeated_label.shape)

    ### split data into 9:1 as num_train_data:num_test_data
    train_data, test_data = np.split(
        repeated_data, [repeated_data.shape[0] * 9 // 10])
    np.save('train_data.npy', train_data)
    np.save('test_data.npy', test_data)
    # print(train_data.shape)
    # print(test_data.shape)
    print('data saved')
    train_label, test_label = np.split(
        repeated_label, [repeated_label.shape[0] * 9 // 10])
    np.save('train_label.npy', train_label)
    np.save('test_label.npy', test_label)
    # print(train_label.shape)
    # print(test_label.shape)
    print('label saved')

    ### read missing data
    missing_mask_filename = DATA_PATH + 'mask_list.json'
    with codecs.open(missing_mask_filename, 'r', 'utf-8') as f:
        miss_mask = json.load(f)
    miss_mask_data = []
    for idx in x_base_list:
        miss_mask_data.append(miss_mask[idx])
    for idx in y_base_list:
        miss_mask_data.append(miss_mask[idx])
    miss_mask_data = np.array(miss_mask_data)
    np.save('miss_mask.npy', miss_mask_data)
    # print(miss_mask_data.shape)
    print('miss_mask saved')


if __name__ == '__main__':
    main()
