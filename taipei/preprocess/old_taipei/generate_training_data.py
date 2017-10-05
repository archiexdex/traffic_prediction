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
import datetime
import time
import math
import matplotlib.pyplot as plt


is_log = 0
data_completion = '_train_50_label_100'

DATA_PATH = "/home/xdex/Desktop/traffic_flow_detection/taipei/training_data/old_Taipei_data/vd_base/"
TOLERANCE = 0
START_TIME = time.mktime( datetime.datetime.strptime("2015-12-01 00:00:00", "%Y-%m-%d %H:%M:%S").timetuple() )

def get_day_minute(timestamp):
    H = int( datetime.datetime.fromtimestamp(timestamp).timetuple()[3] )
    M = int( datetime.datetime.fromtimestamp(timestamp).timetuple()[4] )
    return H * 60 + M

def main():
    """
    main
    """

    traget_vd_filename   = DATA_PATH + "target_vd_list.json"
    vd_grp_lane_filename = DATA_PATH + "vd_grp_lane.json"

    # load vd list and load vd group lane 
    target_vd_list   = {}
    vd_grp_lane_list = {}
    k = 0
    with codecs.open(traget_vd_filename, 'r', 'utf-8') as fp:
        target_vd_list = json.load(fp)
    with codecs.open(vd_grp_lane_filename, 'r', 'utf-8') as fp:
        vd_grp_lane_list = json.load(fp)

    # gather train data into one tensor according to vd_grp_lane_list
    train_data = []
    train_mask = []
    for vd in target_vd_list["train"]:
        for grp in vd_grp_lane_list[vd]:
            
            # Show train VD order
            if is_log == 1:
                print(k, vd, grp)
                k += 1
            vd_filenme       = DATA_PATH + "fix_data/"     + vd + "_" + grp + ".npy"
            mask_filename    = DATA_PATH + "mask_data/"    + vd + "_" + grp + ".npy"
            outlier_filename = DATA_PATH + "mask_outlier/" + vd + "_" + grp + ".npy"

            vd_file      = np.load(vd_filenme)
            mask_file    = np.load(mask_filename)
            outlier_file = np.load(outlier_filename)

            # change time form
            for idx in range(vd_file.shape[0]):
                vd_file[idx][0] = get_day_minute(vd_file[idx][0])

            train_data.append(vd_file)
            mask_file |= outlier_file
            train_mask.append(mask_file)

    # gather label data into one tensor according to vd_grp_lane_list
    label_data = []
    label_mask = []
    k = 0
    for vd in target_vd_list["label"]:
        for grp in vd_grp_lane_list[vd]:
            
            # Show label VD order
            if is_log == 1:
                print(k, vd, grp)
                k += 1
            vd_filenme       = DATA_PATH + "fix_data/"     + vd + "_" + grp + ".npy"
            mask_filename    = DATA_PATH + "mask_data/"    + vd + "_" + grp + ".npy"
            outlier_filename = DATA_PATH + "mask_outlier/" + vd + "_" + grp + ".npy"

            vd_file      = np.load(vd_filenme)
            mask_file    = np.load(mask_filename)
            outlier_file = np.load(outlier_filename)

            # change time form
            for idx in range(vd_file.shape[0]):
                vd_file[idx][0] = get_day_minute(vd_file[idx][0])
            
            label_data.append(vd_file)
            mask_file |= outlier_file
            label_mask.append(mask_file)
    
    # concatenate data and mask
    for idx, (item1, item2) in enumerate(zip(train_data, train_mask)):
        for jdx, (jtem1, jtem2) in enumerate(zip(item1, item2)):
            jtem1[5] = jtem2
    
    for idx, (item1, item2) in enumerate(zip(label_data, label_mask)):
        for jdx, (jtem1, jtem2) in enumerate(zip(item1, item2)):
            jtem1[5] = jtem2

    train_data = np.array(train_data)
    train_mask = np.array(train_mask)
    label_data = np.array(label_data)
    label_mask = np.array(label_mask)

    print(train_data.shape)
    print(train_mask.shape)
    print(label_data.shape)
    print(label_mask.shape)

    # Calculate mean and variable
    tmp_train = []
    for idx in range(train_data.shape[1]):
        if len( np.argwhere(train_mask[:, idx] == 1) ) == 0 :
            tmp_train.append(train_data[:, idx, :])
    tmp_train = np.array(tmp_train)
    tmp_label = []
    for idx in range(label_data.shape[1]):
        if len( np.argwhere(label_mask[:, idx] == 1) ) == 0 :
            tmp_label.append(label_data[:, idx, :])
    tmp_label = np.array(tmp_label)
    
    print(tmp_train.shape)
    print(tmp_label.shape)
    data_normalization(tmp_train, "train")
    data_normalization(tmp_label, "valid")
    
    # Build train data and label data
    input_organized_data = []
    label_organized_data = []
    label_mask_organized_data = []
    vd_time_list = np.zeros( 1440 )
    for i in range(train_data.shape[1] - 12 - 4):

        train = np.argwhere(train_mask[:, i:i + 12] == 1)
        label = np.argwhere(label_mask[:, i + 12:i + 12 + 4] == 1)
        # train_tmp = [[],[],[],[],[],[],
        #             [],[],[],[],[],[],]
        # for _, item in enumerate(train):
        #     item = list(item)
        #     train_tmp[ item[1] ].append(item)
        # label_tmp = [[]]
        # for _, item in enumerate(label):
        #     item = list(item)
        #     label_tmp[ item[1] ].append(item)
        
        # train_size = 0
        # for _, item in enumerate(train_tmp):
        #     if len(item) > 0:
        #         train_size += 1
        # label_size = 0
        # for _, item in enumerate(label_tmp):
        #     if len(item) > 0:
        #         label_size += 1
        # if train_size <= 0 and label_size <= 0:
        if len(train) <= ( train_mask.shape[0] * 12 * 0.5 ) and len(label) <= 0:
        # if len(label) <= 0:
            input_organized_data.append(train_data[:, i:i + 12, :])
            label_organized_data.append(label_data[:, i + 12: i+12 + 4, :])
            
            vd_time_list[int(train_data[0, i, :][0])] += 1
            

    input_organized_data = np.array(input_organized_data)
    label_organized_data = np.array(label_organized_data)
    np.save(DATA_PATH + "train_vd_time_list", vd_time_list)
    

    print("total data shape")
    print(input_organized_data.shape)
    print(label_organized_data.shape)
    
    del train_data
    del train_mask
    del label_data
    del label_mask

    # split data into 9:1 as num_train_data:num_test_data
    train_data, test_data = np.split(
        input_organized_data, [input_organized_data.shape[0] * 9 // 10])

    np.save(DATA_PATH + 'train_data'  + data_completion + '.npy', train_data)
    np.save(DATA_PATH + 'test_data'   + data_completion + '.npy', test_data)
    print(train_data.shape)
    print(test_data.shape)
    print('data saved')
    train_label, test_label = np.split(
        label_organized_data, [label_organized_data.shape[0] * 9 // 10])
    # _, test_mask, _ = np.split(
    #     label_mask_organized_data, [label_mask_organized_data.shape[0] * 8 // 10, label_mask_organized_data.shape[0] * 9 // 10])

    # change time in order to draw data easily
    # test_label[:, :, 0] = test_label[:, :, 0] * 300 + START_TIME   

    np.save(DATA_PATH + 'train_label' + data_completion + '.npy', train_label)
    np.save(DATA_PATH + 'test_label'  + data_completion + '.npy', test_label)
    # np.save(DATA_PATH + 'test_mask.npy', test_mask)

    print(train_label.shape)
    print(test_label.shape)
    print('label saved')


if __name__ == '__main__':
    main()
