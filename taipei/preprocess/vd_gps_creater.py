import os
import csv
import sys
import time
import datetime
import numpy as np

"""
    Variable
"""
read_path = "/home/xdex/Documents/Taipei_xml_data/"
save_path = "/home/xdex/Desktop/traffic_flow_detection/taipei/training_data/new_raw_data/"
file_name = "VD_GPS"
special_vd = ["V9011A0", "V6170C0", "VXPHLE0", "VXPHLA0"]

fd = open(read_path + file_name + ".csv")
csv_cursor = csv.reader(fd)

key = ""
x = y = 0

data = {}

# ['路口編號', '路口名稱', '道路', '大街廓', '方向', '實際車道數', '偵測總車道數', '偵測車道數', '偵測型式', '平面/高架', '所在位置', '道路類型', '小街廓', '偵測器位置', '速限', '地區分類', '所屬轄區分局', '設備狀況', '行政區', 'WGSY', 'WGSX', '']
#     0          1         2       3       4         5             6           7          8           9         10        11         12        13       14         15          16          17        18      19       20
for i, item in enumerate(csv_cursor):
    
    if i == 0:
        print(item)
        continue
    
    key = item[0]
    if item[19] != "" and item[20] != "":
        x = float(item[19])
        y = float(item[20])
        data[key] = [x, y]
    print(i)


print(len(data))    
np.save(save_path + file_name, data)
