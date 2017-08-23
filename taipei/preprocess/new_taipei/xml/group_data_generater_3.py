import os
import sys
import time
import gzip
import xml.etree.ElementTree as ET
import datetime
import json
import threading
import numpy as np

"""
    Variables
"""

# 2015/12/01 00:00:00 ~ 2017/07/31 23:55:00
start_time = time.mktime( datetime.datetime.strptime("2015-12-01 00:00:00", "%Y-%m-%d %H:%M:%S").timetuple() )
end_time   = time.mktime( datetime.datetime.strptime("2017-08-01 00:00:00", "%Y-%m-%d %H:%M:%S").timetuple() )
data_size  = int((end_time - start_time) / 300)

st_time = datetime.datetime.now()

root_path = "/home/xdex/Desktop/traffic_flow_detection/taipei/training_data/new_raw_data/vd_base/"

data_path = root_path + "5/fix_data_lane/"
mask_path = root_path + "5/mask_lane/"
data_save_path = root_path + "5/fix_data_group/"
mask_save_path = root_path + "5/mask_group/"

"""
    Function
"""
# 0 is Sunday
def get_week(timestamp):
    # Because the timetuple return 0 is Monday, so I add one to change 1 to Monday
    return (datetime.datetime.fromtimestamp(timestamp).timetuple()[6] + 1 ) % 7

def get_hour(timestamp):
    # timetuple : year month day hour minute second week ...
    return datetime.datetime.fromtimestamp(timestamp).timetuple()[3] 

vd_grp_lane = {}
with open(root_path + "vd_grp_lane.json") as fp:
    vd_grp_lane = json.load(fp)

flg = False
for vd in vd_grp_lane:
    print(vd)

    for grp in vd_grp_lane[vd]:
        data = []
        mask = []
        grp_size = len(vd_grp_lane[vd][grp])
        for lane in vd_grp_lane[vd][grp]:
            read_data_path = data_path + vd + "_" + lane + ".npy"
            read_mask_path = mask_path + vd + "_" + lane + ".npy"
            tmp_data = []
            tmp_mask = []
            try:
                tmp_data = np.load(read_data_path)
                tmp_mask = np.load(read_mask_path)
            except:
                flg = True
                break

            if data == []:
                data = tmp_data
                mask = tmp_mask
            else:
                data += tmp_data
                mask |= tmp_mask
        
        if flg:
            flg = False
            print("Don't exist", vd)
            break
        # timestamp density flow speed week
        #     0        1      2     3   4
        data = np.array(data)
        # print(data.shape)
        data[:,0:0+2] /= grp_size
        data[:,3:3+2] /= grp_size
        np.save(data_save_path + vd + "_" + grp , data)
        np.save(mask_save_path + vd + "_" + grp , mask)
        

ed_time = datetime.datetime.now()
print("Finishing...", ed_time-st_time)