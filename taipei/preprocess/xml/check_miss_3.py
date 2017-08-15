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
long_period = 4
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

miss_dict = {}

def check_data(path):
    data = np.load(path)
    t = np.argwhere(data == 1)
    miss_dict[vd_name] = len(t) / data.shape[0]


for root, dirs, files in os.walk(mask_path):
    for file in files:
        path = os.path.join(mask_path, file)
        print("Fixing VD: "+file)
        vd_name = file[:7]
        check_data(path)
        
    break

with open(root_path + "5/miss_rate.json", 'w') as fp:
    json.dump(miss_dict, fp)