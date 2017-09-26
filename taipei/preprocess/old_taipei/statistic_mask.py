"""
statistic mask for each vd
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
from functools import reduce


is_log = 0

DATA_PATH = "/home/xdex/Desktop/traffic_flow_detection/taipei/training_data/old_Taipei_data/vd_base/"
TOLERANCE = 0
START_TIME = time.mktime( datetime.datetime.strptime("2015-12-01 00:00:00", "%Y-%m-%d %H:%M:%S").timetuple() )
st_time = datetime.datetime.now()

def main():
    """
    main
    """

    traget_vd_filename   = DATA_PATH + "target_vd_list.json"
    vd_grp_lane_filename = DATA_PATH + "vd_grp_lane.json"

    # load vd list and load vd group lane 
    target_vd_list   = {}
    vd_grp_lane_list = {}
    with codecs.open(traget_vd_filename, 'r', 'utf-8') as fp:
        target_vd_list = json.load(fp)
    with codecs.open(vd_grp_lane_filename, 'r', 'utf-8') as fp:
        vd_grp_lane_list = json.load(fp)

    # gather train data into one tensor according to vd_grp_lane_list
    train_data = []
    train_mask = []
    train_vd_list = {"train":[], "label":[]}
    k = 0
    for vd in target_vd_list["train"]:
        for grp in vd_grp_lane_list[vd]:
            
            # Show train VD order
            train_vd_list["train"].append(vd+"_"+grp)
            if is_log == 1:
                print(k, vd, grp)
                k += 1
            
            vd_filenme       = DATA_PATH + "fix_data/"     + vd + "_" + grp + ".npy"
            mask_filename    = DATA_PATH + "mask_data/"    + vd + "_" + grp + ".npy"
            outlier_filename = DATA_PATH + "mask_outlier/" + vd + "_" + grp + ".npy"

            vd_file      = np.load(vd_filenme)
            mask_file    = np.load(mask_filename)
            outlier_file = np.load(outlier_filename)

            vd_file[:,0] = (vd_file[:,0] - START_TIME) / 300
            vd_file[:,0] %= 1440

            train_data.append(vd_file)
            mask_file |= outlier_file
            train_mask.append(mask_file)

    train_data = np.array(train_data)
    train_mask = np.array(train_mask)

    print(train_data.shape)
    print(train_mask.shape)

    # concatenate data and mask
    for idx, (item1, item2) in enumerate(zip(train_data, train_mask)):
        for jdx, (jtem1, jtem2) in enumerate(zip(item1, item2)):
            jtem1[5] = jtem2

    with open(DATA_PATH + "train_vd_list.json", "w") as fp:
        json.dump(train_vd_list, fp)

    vd_missing_rate = { }
    for i in range(10):
        vd_missing_rate[i] = 0
    
    for idx, item in enumerate(train_mask):
        s = reduce((lambda x, y: x + y), item)
        rate = s / len(item)
        vd_missing_rate[train_vd_list["train"][idx]] = rate
        if rate > 0.1:
            print(train_vd_list["train"][idx])
        
        if rate > 0.9:
            vd_missing_rate[9] += 1
        elif rate > 0.8:
            vd_missing_rate[8] += 1
        elif rate > 0.7:
            vd_missing_rate[7] += 1
        elif rate > 0.6:
            vd_missing_rate[6] += 1
        elif rate > 0.5:
            vd_missing_rate[5] += 1
        elif rate > 0.4:
            vd_missing_rate[4] += 1
        elif rate > 0.3:
            vd_missing_rate[3] += 1
        elif rate > 0.2:
            vd_missing_rate[2] += 1
        elif rate > 0.1:
            vd_missing_rate[1] += 1
        else:
            vd_missing_rate[0] += 1

    with open(DATA_PATH + "vd_missing_rate.json", "w") as fp:
        json.dump(vd_missing_rate, fp)
    
    # exit()
    vd_time_list = np.zeros(( int(train_data.shape[1]/(12*24)),train_data.shape[0],288), dtype=int)
    for idx, item in enumerate(train_data):
        for jdx, jtem in enumerate(item):
            # time, density, flow, speed, week, mask, timestamp
            now = jtem[6]
            H = int( datetime.datetime.fromtimestamp(now).timetuple()[3] )
            M = int( datetime.datetime.fromtimestamp(now).timetuple()[4] )

            vd_time_list[int(jdx/(12*24) )][idx][H * 12 + int(M // 5)] += train_mask[idx][jdx]
            
        
    np.save(DATA_PATH + "vd_time_mask_alltime.npy", vd_time_list)
    ed_time = datetime.datetime.now()
    print("Finishing...", ed_time - st_time)

if __name__ == '__main__':
    main()
