import os
import sys
import time
import gzip
import xml.etree.ElementTree as ET
from datetime import datetime
import json
import threading
import numpy as np

start_time = datetime.now()
# st_time = time.mktime( datetime.strptime("2015-12-01 00:05:00", "%Y-%m-%d %H:%M:%S").timetuple() )
save_path = "/home/xdex/Desktop/traffic_flow_detection/taipei/training_data/new_raw_data/time_base/"
use_thread = 0


def parse_data(root_path):
    
    data_1 = {}
    data_5 = {}
    density = 0
    flow = 0
    speed = 0
    time_stamp = 0
    key = ""
    lane_order = 0
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
            if f[8] == "_":
                continue
            
            xml = gzip.open(path, "r") 
            tree = ET.ElementTree(file=xml)

            for elem in tree.iter():
                # print(elem.tag, elem.attrib)
                # print(data_1)
                # print(data_5)
                # input()
                # If the VD is not in dictionary, create a new key for the VD and append data into dictionary
                # There may exists another time stamp, so store the data as type list
                if elem.tag == "Info":
                    key        = elem.attrib["vdid"][5:]
                    time_stamp = int(time.mktime( datetime.strptime(elem.attrib["datacollecttime"], "%Y/%m/%d %H:%M:%S").timetuple() ))

                if elem.tag == "lane":
                    lane_order = int(elem.attrib["vsrdir"])
                    density    = float(elem.attrib["laneoccupy"])
                    speed      = float(elem.attrib["speed"])
                
                if elem.tag == "cars":
                    flow += int(elem.attrib["volume"])
                    flg  += 1
                    # Store data
                    if flg == 3:
                        # For 1 minute data
                        if f[8] == "_":
                            if key not in data_1:
                                data_1[key] = {}
                            
                            if lane_order not in data_1[key]:
                                data_1[key][lane_order] = {}
                            
                            if time_stamp not in data_1[key][lane_order]:
                                data_1[key][lane_order][time_stamp] = [{"density": density, "flow": flow, "speed": speed, "time": time_stamp}]
                            else:
                                data_1[key][lane_order][time_stamp].append( {"density": density, "flow": flow, "speed": speed, "time": time_stamp} )
                        # For 5 minute data
                        if f[8] == "5":
                            if key not in data_5:
                                data_5[key] = {}
                            
                            if lane_order not in data_5[key]:
                                data_5[key][lane_order] = {}
                            
                            if time_stamp not in data_5[key][lane_order]:
                                data_5[key][lane_order][time_stamp] = [{"density": density, "flow": flow, "speed": speed, "time": time_stamp}]
                            else:
                                data_5[key][lane_order][time_stamp].append( {"density": density, "flow": flow, "speed": speed, "time": time_stamp} )
                        
                        # Initialize Variables
                        flg = 0
                        flow = 0
            xml.close()

    print("data1 length:", len(data_1))
    print("data5 length:", len(data_5))
    with open(save_path + "1/" + save_name + ".json", 'w') as fp:
        json.dump(data_1, fp)
    with open(save_path + "5/" + save_name + ".json", 'w') as fp:
        json.dump(data_5, fp)

root_path = "/home/xdex/Documents/Taipei_xml_data/XML/"

class MyThread(threading.Thread):
    def __init__(self, thread_id, thread_name, dir_list):
        threading.Thread.__init__(self)
        self.thread_id   = thread_id
        self.thread_name = thread_name
        self.dir_list    = dir_list

    def run(self):
        for d in self.dir_list:
            path = os.path.join(root_path, d)
            print(self.thread_name ,path)
            parse_data(path)

dir_list = []
for root, dirs, files in os.walk(root_path):
    dir_list = dirs
    break

if use_thread == 0:
    for d in dir_list:
        path = os.path.join(root_path, d)
        print(path)
        parse_data(path)

if use_thread == 1:
    dir_size = len(dir_list)
    # Initialize new threads
    thread1 = MyThread(1, "Thread-1", dir_list[dir_size//8 * 0:dir_size//8 * 2])
    thread2 = MyThread(2, "Thread-2", dir_list[dir_size//8 * 2:dir_size//8 * 4])
    thread3 = MyThread(3, "Thread-3", dir_list[dir_size//8 * 4:dir_size//8 * 6])
    thread4 = MyThread(4, "Thread-4", dir_list[dir_size//8 * 6:])

    # thread5 = MyThread(5, "Thread-5", dir_list[dir_size//8 * 4:dir_size//8 * 5])
    # thread6 = MyThread(6, "Thread-6", dir_list[dir_size//8 * 5:dir_size//8 * 6])
    # thread7 = MyThread(7, "Thread-7", dir_list[dir_size//8 * 6:dir_size//8 * 7])
    # thread8 = MyThread(8, "Thread-8", dir_list[dir_size//8 * 7:])

    thread_list = [ thread1, thread2, thread3, thread4]
    # Start threads
    for i in thread_list:
        i.start()

    # Join threads
    for i in thread_list:
        i.join()
end_time = datetime.now()
print("Finish...", (end_time-start_time))