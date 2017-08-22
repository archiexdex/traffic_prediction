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
# choose 1 or 0 to fix different mode data
mode = 0
# 2015/12/01 00:00:00 ~ 2017/07/31 23:55:00
start_time = time.mktime( datetime.datetime.strptime("2015-12-01 00:00:00", "%Y-%m-%d %H:%M:%S").timetuple() )
end_time   = time.mktime( datetime.datetime.strptime("2017-08-01 00:00:00", "%Y-%m-%d %H:%M:%S").timetuple() )
data_size  = int((end_time - start_time) / 300)

st_time = datetime.datetime.now()

root_path = "/home/xdex/Desktop/traffic_flow_detection/taipei/training_data/new_raw_data/vd_base/"
data_path = ""
mask_path = ""
save_path = ""
vd_name   = "" 
if mode == 0:
    data_path = root_path + "5/data_group/"
    mask_path = root_path + "5/mask_group/" 
    save_path = root_path + "5/fix_data_group/"
elif mode == 1:
    data_path = root_path + "5/data_lane/"
    mask_path = root_path + "5/mask_lane/" 
    save_path = root_path + "5/fix_data_lane/"

miss_dict = {}
for i in range(10):
    miss_dict[i] = 0
max_miss = -1023456789
min_miss =  1023456789

def check_data(path):
    global max_miss
    global min_miss

    data = np.load(path)
    t = np.argwhere(data == 1)
    miss_rate = len(t) / data.shape[0]
    if vd_name not in miss_dict:
        miss_dict[vd_name] = {}
    miss_dict[vd_name][vd_grp] = miss_rate

    max_miss = miss_rate if max_miss < miss_rate else max_miss
    min_miss = miss_rate if min_miss > miss_rate else min_miss
    miss_dict["max_miss"] = max_miss
    miss_dict["min_miss"] = min_miss
    
    if miss_rate > 0.9:
        miss_dict[9] += 1
    elif miss_rate > 0.8:
        miss_dict[8] += 1
    elif miss_rate > 0.7:
        miss_dict[7] += 1
    elif miss_rate > 0.6:
        miss_dict[6] += 1
    elif miss_rate > 0.5:
        miss_dict[5] += 1
    elif miss_rate > 0.4:
        miss_dict[4] += 1
    elif miss_rate > 0.3:
        miss_dict[3] += 1
    elif miss_rate > 0.2:
        miss_dict[2] += 1
    elif miss_rate > 0.1:
        miss_dict[1] += 1
    else:
        miss_dict[0] += 1
    

for root, dirs, files in os.walk(mask_path):
    for file in files:
        path = os.path.join(mask_path, file)
        print("Fixing VD: "+file)
        vd_name = file[:7]
        vd_grp = file[8:9]
        check_data(path)
        
    break
if mode == 0:
    with open(root_path + "5/miss_rate_group.json", 'w') as fp:
        json.dump(miss_dict, fp)
elif mode == 1:
    with open(root_path + "5/miss_rate_lane.json", 'w') as fp:
        json.dump(miss_dict, fp)


ed_time = datetime.datetime.now()
print("Finishing...", ed_time-st_time)