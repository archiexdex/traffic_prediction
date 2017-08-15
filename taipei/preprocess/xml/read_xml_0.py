import os
import sys
import time
import gzip
import xml.etree.ElementTree as ET
from datetime import datetime
import numpy as np
import json


# st_time = time.mktime( datetime.strptime("2015-12-01 00:05:00", "%Y-%m-%d %H:%M:%S").timetuple() )
save_path = "/home/xdex/Desktop/traffic_flow_detection/taipei/training_data/new_raw_data/"

def parse_data(root_path):
    
    data_1 = {}
    data_5 = {}
    density = 0
    flow = 0
    speed = 0
    time_stamp = 0
    key = ""
    flg = 0

    save_name = root_path[-8:]
    

    # 無關     XML資料間隔 XML更新時間 更新間隔   63000+VD編號 無關    偵測時間         無關    車道   車速   佔有率     車輛種類  車流量  車輛種類: L 大車 S 小型車 M 機車 
    # version listname   updatetime interval  vdid        status datacollecttime vsrdir vsrid speed laneoccupy carid   volume
    for r, _, files in os.walk(root_path):
        
        for f in files:
            # print(f, f[8])
            path = os.path.join(root_path, f)

            if f[3:3+4] == "info":
                continue
            
            xml = gzip.open(path, "r") 
            tree = ET.ElementTree(file=xml)

            for elem in tree.iter():
                # print (elem.tag, elem.attrib, data)
                # input()
                if elem.tag == "Info":
                    
                    # Store data we read before
                    if flg > 0:
                        # If the VD is not in dictionary, create a new key for the VD and append data into dictionary
                        # There may exists another time stamp, so store the data as type list

                        if f[8] == "_":
                            if key not in data_1:
                                data_1[key] = {}
                                data_1[key][time_stamp] = [{"density": density, "flow": flow, "speed": speed, "time": time_stamp}]

                            else:
                                if time_stamp not in data_1[key]:
                                    data_1[key][time_stamp] = [{"density": density, "flow": flow, "speed": speed, "time": time_stamp}]
                                else:
                                    data_1[key][time_stamp].append( {"density": density, "flow": flow, "speed": speed, "time": time_stamp} )
                        
                        if f[8] == "5":
                            if key not in data_5:
                                data_5[key] = {}
                                data_5[key][time_stamp] = [{"density": density, "flow": flow, "speed": speed, "time": time_stamp}]
                            
                            else:
                                if time_stamp not in data_5[key]:
                                    data_5[key][time_stamp] = [{"density": density, "flow": flow, "speed": speed, "time": time_stamp}]
                                else:
                                    data_5[key][time_stamp].append( {"density": density, "flow": flow, "speed": speed, "time": time_stamp} )
                        
                        # Initialize variable
                        density = 0
                        flow = 0
                        speed = 0
                        time_stamp = 0
                        key = ""
                        flg = 0

                    # Start store for new data
                    key        = elem.attrib["vdid"][5:]
                    time_stamp = int(time.mktime( datetime.strptime(elem.attrib["datacollecttime"], "%Y/%m/%d %H:%M:%S").timetuple() ))
                    flg += 1

                if elem.tag == "lane":
                    density = float(elem.attrib["laneoccupy"])
                    speed   = float(elem.attrib["speed"])
                
                if elem.tag == "cars":
                    flow += int(elem.attrib["volume"])

            xml.close()

    print("data1 length:", len(data_1))
    print("data5 length:", len(data_5))
    with open(save_path + "1/" + save_name + ".json", 'w') as fp:
        json.dump(data_1, fp)
    with open(save_path + "5/" + save_name + ".json", 'w') as fp:
        json.dump(data_5, fp)


root_path = "/home/xdex/Documents/Taipei_xml_data/XML/"


dir_list = []

for root, dirs, files in os.walk(root_path):
    dir_list = dirs
    break

for d in dir_list:
    # if d != "20151201":
    #     continue
    path = os.path.join(root_path, d)
    print(path)
    parse_data(path )
