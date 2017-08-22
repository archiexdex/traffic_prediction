import os
import csv
import time
import gzip
import xml.etree.ElementTree as ET
import datetime
import json
import threading
import numpy as np

start_time = datetime.datetime.now()
save_path = "/home/xdex/Desktop/traffic_flow_detection/taipei/training_data/new_raw_data/vd_base/"
use_thread = 0

"""
    Function
"""
def parse_data(root_path):
    
    data = {}
    
    # DEVICEID	LANEGROUPID	LANEGROUPNAME	LANEID	ALIAS	RVSLGID	RVSLGNAME
    #    0          1             2            3      4        5        6

    with open(root_path) as fp:
        csv_cursor = csv.reader(fp)
        for i, item in enumerate(csv_cursor):
            print(i, item)
            if i == 0:
                continue
            vd = item[0]
            group = item[1]
            lane = item[3]
            
            if vd not in data:
                data[vd] =  {}
            if group not in data[vd]:
                data[vd][group] = []
            data[vd][group].append(lane)
    
    with open(save_path + "vd_grp_lane.json", "w") as fp:
        json.dump(data,fp)

"""
    Main
"""
root_path = "/home/xdex/Documents/Taipei_xml_data/"

parse_data(root_path + "VD_lane_grp.csv")
end_time = datetime.datetime.now()
print("Finish...", (end_time-start_time))