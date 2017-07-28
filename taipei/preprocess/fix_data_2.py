import numpy as np
import os
import json
import time
from datetime import datetime


root_path = "../"

print("Reading VD_GPS...")
vd_gps = np.load(root_path+"VD_GPS.npy").item()
# data = np.load(root_path+"input_data.npy").item()

print("Reading raw data...")
data = {}
with open(root_path+"raw_data.json") as file:
    data = json.load(file)

print("Reading vd_list...")
vd_list = []
with open(root_path+"reduce_dimension.json") as file:
    tmp = json.load(file)
    for i in tmp["x_base"]:
        vd_list.append(i)

# Find max time range
st_time = time.mktime( datetime.strptime("2015-01-01 00:05:00", "%Y-%m-%d %H:%M:%S").timetuple() )
ed_time = time.mktime( datetime.strptime("2017-03-13 00:00:00", "%Y-%m-%d %H:%M:%S").timetuple() )


print(st_time, ed_time)
def get_week(timestamp):
    return (datetime.fromtimestamp(timestamp).timetuple()[6] + 1 ) % 7

def get_date_string(timestamp):
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

# print(datetime.fromtimestamp(st_time).timetuple())
# print(datetime.fromtimestamp(ed_time)) 

# exit()
miss_list = {}
miss_len_list = {}
mask_list = {}
bucket = {}
total_size = int(ed_time - st_time) // 300 + 1

for key in vd_list:
    flg = 0
    miss_list[key] = {}
    miss_len_list[key] = {}
    mask_list[key] = {}
    bucket = {}

    # To get redundant data
    for i, grp in enumerate(data[key]):
        
        bucket[grp] = {}
        for item in data[key][grp]:    
            tid = int(item[4]-st_time) // 300
            
            bucket[grp][tid] = []
            bucket[grp][tid].append(item)
            # print(key, item, tid)
            # input("@")
    
    # Initialize all group in data[key]
    for grp in data[key]:
        data[key][grp] = [0] * total_size

    # To reasign data
    for grp in bucket:
        for i in bucket[grp]:
            
            if len(bucket[grp][i]) > 1:
                bucket[grp][i] = reduce(lambda x, y: x + y, bucket[grp][i]) / len(bucket[grp][i])

            data[key][grp][i] = bucket[grp][i][0]
            

    
    for i, grp in enumerate(data[key]):
        mask_list[key][grp] = []
        miss_list[key][grp] = []
        miss_len_list[key][grp] = 0
        now = st_time
        for item in data[key][grp]:
            if item == 0:
                w = get_week(now)
                mask_list[key][grp].append(1)
                miss_list[key][grp].append(get_date_string(now) )
                data[key][grp][i] = [0, 0, 0, w, now]
            else:
                mask_list[key][grp].append(0)
            now += 300

        print(key, grp, len(data[key][grp]), len(miss_list[key][grp]) )

        miss_len_list[key][grp] = ( len(miss_list[key][grp]) )
    # input("@")        

print("Saving miss_len_data...")
with open(root_path+'miss_len_data.json', 'w') as fp:
    json.dump(miss_len_list, fp)

print("Saving miss_data...")
with open(root_path+'miss_data.json', 'w') as fp:
    json.dump(miss_list, fp)

print("Saving fix_raw_data...")
with open(root_path+'fix_raw_data.json', 'w') as fp:
    json.dump(data, fp)

print("Saving mask_list...")
with open(root_path+'mask_list.json', 'w') as fp:
    json.dump(mask_list, fp)

