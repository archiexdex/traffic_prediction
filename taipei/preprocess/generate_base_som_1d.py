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

DATA_PATH = '/home/xdex/Desktop/traffic_flow_detection/taipei/'
TOLERANCE = 0


def main():
    """
    main
    """

    # load raw data
    raw_filename = DATA_PATH + 'fix_raw_data.json'
    vd_id_filename = DATA_PATH + 'vd_list_som_1.npy'
    
    with codecs.open(raw_filename, 'r', 'utf-8') as f:
        raw_data = json.load(f)
    # print(np.array(list(raw_data.values())).shape)
    # vd_id_filename = DATA_PATH + 'reduce_dimension.json'
    # with open(vd_id_filename) as fp:
    #     vd_list = json.load(fp)
    # vd_list = vd_list['y_base']
    
    # read vd id list
    vd_list = np.load(vd_id_filename)
    # gather data into one tensor according to vd id list
    all_data = []
    for idx in vd_list:
        for grp in raw_data[idx]:
            all_data.append(raw_data[idx][grp])
            # tmp = np.array(raw_data[idx][grp])
            # for i in raw_data[idx][grp]:
            #     print (i)
            #     if type(i) != list:
            #         print(i)

            #         # print(tmp.shape)
            #         # print(tmp.dtype)
            #         input("!")

    all_data = np.array(all_data)
    print(all_data.shape)
    # exit()
    # generate repeared data from shape=[2*vds, intervals, features] to
    # shape=[num_data, 2*vds, 12, features]
    repeated_data = []
    # iter all interval
    for i in range(all_data.shape[1] - 12 - 1):
        # (i + 12 + 1) == (i + ref times 12 + label times 1)
        repeated_data.append(all_data[:, i:i + 12 + 1, :])
    repeated_data = np.array(repeated_data)
    # print(repeated_data.shape)

    # discard missing data
    organized_data = []
    good_list = []
    for i, v in enumerate(repeated_data):
        missing_list = np.logical_and(np.logical_and(
            v[:, :, 0] == 0, v[:, :, 1] == 0), v[:, :, 2] == 0)
        missing_arg_list = np.argwhere(missing_list)
        if len(missing_arg_list) <= TOLERANCE:
            organized_data.append(v)
            good_list.append(i)
        # else:
        #     print(missing_arg_list)
        #     for _, vv in enumerate(missing_arg_list):
        #         print(v[vv[0], vv[1], :3])
    organized_data = np.array(organized_data)
    # print(organized_data.shape)

    input_organized_data = organized_data[:, :, :12, :]
    label_organized_data = organized_data[:, :, 12, 1]
    print(input_organized_data.shape)
    print(label_organized_data.shape)

    # split data into 9:1 as num_train_data:num_test_data
    train_data, test_data = np.split(
        input_organized_data, [input_organized_data.shape[0] * 9 // 10])
    np.save('train_data.npy', train_data)
    np.save('test_data.npy', test_data)
    print(train_data.shape)
    print(test_data.shape)
    print('data saved')
    train_label, test_label = np.split(
        label_organized_data, [label_organized_data.shape[0] * 9 // 10])
    np.save('train_label.npy', train_label)
    np.save('test_label.npy', test_label)
    print(train_label.shape)
    print(test_label.shape)
    print('label saved')

    # read missing data
    # missing_mask_filename = DATA_PATH + 'mask_list.json'
    # with codecs.open(missing_mask_filename, 'r', 'utf-8') as f:
    #     miss_mask = json.load(f)
    # miss_mask_data = []
    # for idx in x_base_list:
    #     miss_mask_data.append(miss_mask[idx])
    # for idx in y_base_list:
    #     miss_mask_data.append(miss_mask[idx])
    # miss_mask_data = np.array(miss_mask_data)
    # np.save('miss_mask.npy', miss_mask_data)
    # print(miss_mask_data.shape)
    # print('miss_mask saved')


if __name__ == '__main__':
    main()
