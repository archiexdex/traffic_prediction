from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import datetime
import time
import heapq
import json
import numpy as np

"""
    Variables
"""
# choose 1 or 5 to fix different mode data
mode = 5
time_padding = 30
# 1 means time = mode * 1 e.g. long_period = 1 and mode = 5 means if there are more than 5 minutes data be [0, 0, 0], we will mask it
long_period = 6
# 2015/12/01 00:00:00 ~ 2017/07/31 23:55:00
start_time = time.mktime( datetime.datetime.strptime("2015-12-01 00:00:00", "%Y-%m-%d %H:%M:%S").timetuple() )
end_time   = time.mktime( datetime.datetime.strptime("2017-08-01 00:00:00", "%Y-%m-%d %H:%M:%S").timetuple() )
data_size  = int((end_time - start_time) / 300)

root_path = "/home/xdex/Desktop/traffic_flow_detection/taipei/training_data/new_raw_data/vd_base/"
data_path = ""
mask_path = ""
save_path = ""
vd_name   = "" 
if mode == 1:
    data_path = root_path + "1/data/"
    mask_path = root_path + "1/mask/"
    save_path = root_path + "1/fix_data/"
elif mode == 5:
    data_path = root_path + "5/data/"
    mask_path = root_path + "5/mask/" 
    save_path = root_path + "5/fix_data/"

"""
    Function
"""
# 0 is Sunday
def get_week(timestamp):
    # Because the timetuple return 0 is Monday, so I add one to change 1 to Monday
    return (datetime.datetime.fromtimestamp(timestamp).timetuple()[6] + 1 ) % 7

# Save mask and fix_data
def check_data(path):
    data = np.load(path)
    fix_data = [0] * data_size
    mask_list = [0] * data_size
    now = start_time
    ptr = 0
    i = 0
    count = 0
    
    while True:
        # item : [time, density, flow, speed, week]
        item = []
        if i < data.shape[0]:
            item = data[i].tolist() 
        
        if now >= end_time:
            break
        
        # It may occur when data has duplicate time or begging time is lower than start_time
        if i < data.shape[0] and now > item[0] + time_padding:
            i += 1
            continue
        # It may occur when data has missing data
        elif i >= data.shape[0] or item[0] > now + time_padding:
            fix_data[ptr] = [now, 0, 0, 0, get_week(now)]
            mask_list[ptr] = 1
            ptr += 1
            
        else:
            # To check if [density, flow, speed] is [0, 0, 0] and continue in long_period, then they are missing data
            if item[1:1+3] == [0, 0, 0]:
                count += 1
            else:    
                if count > long_period:
                    for k in range(count):
                        mask_list[ptr-k] = 1
                    count = 0
            
            fix_data[ptr] = item
            ptr += 1
            i   += 1

        if mode == 1:
            now += 60
        elif mode == 5:
            now += 300 
    # print("Saving fix_data...")
    np.save(save_path + vd_name, fix_data)  
    # print("Saving mask_data...")
    np.save(mask_path + vd_name, mask_list)

"""
    Main
"""
for root, dirs, files in os.walk(data_path):
    for file in files:
        path = os.path.join(data_path, file)
        print("Fixing VD: "+file)
        vd_name = file[:7]
        check_data(path)
        
    break

