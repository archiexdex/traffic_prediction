from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import time
import heapq
import numpy as np
import json

# Variable
root_path  = "/home/xdex/Desktop/traffic_flow_detection/taipei/training_data/new_raw_data/"
read_path = root_path + "time_base/"
mode = 5
start_time = datetime.datetime.now()

# 20151201 - 20170731
def get_date(day):
    return (datetime.date(2015, 12, 1) + datetime.timedelta(days=day)).strftime("%Y%m%d")

# 0 is Sunday
def get_week(timestamp):
    return (datetime.datetime.fromtimestamp(timestamp).timetuple()[6] + 1 ) % 7

i = 0
raw_data = {}
while True:
    
    now = get_date(i)
    if now == "20170801":
        break

    print(now)
    file_path = read_path + "1/" + now + ".json" if mode == 1 else read_path + "5/" + now + ".json"
    
    with open(file_path ) as fp:
        tmp = json.load(fp)
        for idx, vd in enumerate(tmp):
            
            if vd not in raw_data:
                raw_data[vd] = {}

            for jdx, grp in enumerate(tmp[vd]):
                
                if grp not in raw_data[vd]:
                    raw_data[vd][grp] = []
                    
                for kdx, time_stamp in enumerate(tmp[vd][grp]):
                
                    density = 0
                    flow = 0
                    speed = 0
                    w = get_week(int(time_stamp))
                    t = int(time_stamp)
                    for _, item in enumerate(tmp[vd][grp][time_stamp]):
                        density += item["density"]
                        flow += item["flow"]
                        speed += item["speed"]
                    density /= len(tmp[vd][grp][time_stamp])
                    flow /= len(tmp[vd][grp][time_stamp])
                    speed /= len(tmp[vd][grp][time_stamp])
                    data = [t, density, flow, speed, w]
                    raw_data[vd][grp].append(data)
        

    i += 1

# For each VD every lane, save it
for i in raw_data:
    for j in raw_data[i]:
        print("Saving VD: ", i, "with group: ", j)
        raw_data[i][j].sort()
        save_path = root_path + "vd_base/1/data/" + i + "_" + j if mode == 1 else root_path + "vd_base/5/data/" + i + "_" + j
        np.save(save_path, raw_data[i][j])


end_time = datetime.datetime.now()
print("Finish...", end_time-start_time)