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
import sys
import getopt
import matplotlib.pyplot as plt


DATA_PATH = '/home/xdex/Desktop/traffic_flow_detection/taipei/training_data/'
TOLERANCE = 0


def main():
    """
    main
    """

    # load raw data
    raw_filename = DATA_PATH + 'fix_raw_data_time1.json'
    vd_id_filename = DATA_PATH + 'selected_vd.json'
    mask_vd_filename = DATA_PATH + 'mask_list.json'

    with codecs.open(raw_filename, 'r', 'utf-8') as f:
        raw_data = json.load(f)
    # print(np.array(list(raw_data.values())).shape)

    # read vd id list
    vd_list = {}
    with open(vd_id_filename) as fp:
        vd_list = json.load(fp)

    # read mask list
    mask_list = {}
    with open(mask_vd_filename) as fp:
        mask_list = json.load(fp)

    # gather train data into one tensor according to vd id list
    train_data = []
    train_mask = []
    for key in vd_list["train"]:
        for grp in raw_data[key]:
            train_data.append(raw_data[key][grp])

        for grp in mask_list[key]:
            train_mask.append(mask_list[key][grp])

    # gather label data into one tensor according to vd id list
    label_data = []
    label_mask = []
    for i in vd_list["label"]:
        if i == "0":
            for key in vd_list["label"][i]:
                for grp in raw_data[key]:
                    label_data.append(raw_data[key][grp])

                for grp in mask_list[key]:
                    label_mask.append(mask_list[key][grp])

    train_data = np.array(train_data)
    train_mask = np.array(train_mask)
    label_data = np.array(label_data)
    label_mask = np.array(label_mask)

    print(train_data.shape)
    print(train_mask.shape)
    print(label_data.shape)
    print(label_mask.shape)

    input_organized_data = []
    label_organized_data = []
    for i in range(train_data.shape[1] - 12):

        train = np.argwhere(train_mask[:, i:i + 12] == 1)
        label = np.argwhere(label_mask[:, i + 12:i + 13] == 1)

        if len(train) == 0 and len(label) == 0:
            input_organized_data.append(train_data[:, i:i + 12, :])
            label_organized_data.append(label_data[:, i + 12, 1])

    input_organized_data = np.array(input_organized_data)
    label_organized_data = np.array(label_organized_data)

    print(input_organized_data.shape)
    print(label_organized_data.shape)

    del train_data
    del train_mask
    del label_data
    del label_mask

    # split data into 9:1 as num_train_data:num_test_data
    train_data, test_data = np.split(
        input_organized_data, [input_organized_data.shape[0] * 9 // 10])
    np.save(DATA_PATH + 'train_data.npy', train_data)
    np.save(DATA_PATH + 'test_data.npy', test_data)
    print(train_data.shape)
    print(test_data.shape)
    print('data saved')
    train_label, test_label = np.split(
        label_organized_data, [label_organized_data.shape[0] * 9 // 10])
    np.save(DATA_PATH + 'train_label.npy', train_label)
    np.save(DATA_PATH + 'test_label.npy', test_label)
    print(train_label.shape)
    print(test_label.shape)
    print('label saved')


if __name__ == '__main__':
    main()
