import numpy as np
import os
import json
import time
from datetime import datetime

process_st = datetime.now()

root_path = "/home/xdex/Desktop/traffic_flow_detection/taipei/training_data/old_Taipei_data/vd_base/"
RAW_DATA_PATH = "/home/xdex/Desktop/traffic_flow_detection/taipei/training_data/old_Taipei_data/vd_base/raw_data/"
FIX_DATA_PATH = "/home/xdex/Desktop/traffic_flow_detection/taipei/training_data/old_Taipei_data/vd_base/fix_data/"
MASK_DATA_PATH = "/home/xdex/Desktop/traffic_flow_detection/taipei/training_data/old_Taipei_data/vd_base/mask_data/"

time_period = 6

print("Reading raw data...")
data = {}
# with open(root_path+"raw_data.json") as file:
#     data = json.load(file)

print("Reading vd_list...")
vd_list = []
with open(root_path + "target_vd_list.json") as file:
    tmp = json.load(file)
    vd_list = tmp["total"]

vd_grp_lane_list = {}
with open(root_path + "vd_grp_lane.json") as fp:
    vd_grp_lane_list = json.load(fp)

# Find max time range
st_time = time.mktime( datetime.strptime("2015-01-01 00:05:00", "%Y-%m-%d %H:%M:%S").timetuple() )
ed_time = time.mktime( datetime.strptime("2017-03-13 00:00:00", "%Y-%m-%d %H:%M:%S").timetuple() )

# 0 is Sunday
def get_week(timestamp):
    # Because the timetuple return 0 is Monday, so I add one to change 1 to Monday
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

# Enumerate all VD 
for vd in vd_list:
    
    for grp in vd_grp_lane_list[vd]:
        
        target_name = vd + "_" + str(grp) + ".npy"
        print(target_name)
        data = []
        try:
            data = np.load( RAW_DATA_PATH + target_name)
        except:
            print(target_name, " not exist!! QQ")
            continue

        bucket = {}
        for item in data:    
            tid = int(item[0]-st_time) // 300
            
            bucket[tid] = []
            bucket[tid].append(item)
        
        fix_data = [0] * total_size
                
        for idx, item in enumerate(bucket):
            # print(item, idx, bucket[item])
            if len( bucket[item] ) > 1:
                bucket[item] = reduce(lambda x, y: x + y, bucket[item]) / len(bucket[item])
            
            fix_data[item] = bucket[item][0]

        now = st_time
        mask_data = np.zeros(total_size, dtype=int)
        for idx, item in enumerate(fix_data):
            
            # print(item, now, mask_data[idx])
            if type(item) == int:
                # print("!")
                fix_data[idx] = [now, 0, 0, 0, get_week(now), 1, now]
                mask_data[idx] = 1
            
            now += 300
        
        count = 0
        for ptr, item in enumerate(fix_data):
            
            if item[0] == 0 and item[1] == 0 and item[2] == 0:
                
                count += 1
            else :
                if count >= time_period:
                    for k in range(count):
                        mask_data[ptr-k-1] = 1
                        fix_data[idx][5] = 1
                # print(item, count, mask_list[vd][grp][ptr])
                # input("!")
                count = 0
                    
        # print("finish finding 30 (0,0,0)s")
        np.save( FIX_DATA_PATH + target_name, fix_data)
        np.save( MASK_DATA_PATH + target_name, mask_data)

    # To get redundant data and put them into correspondant bucket
    # for i, grp in enumerate(data[key]):
        
    #     bucket[grp] = {}
    #     for item in data[key][grp]:    
    #         tid = int(item[4]-st_time) // 300
            
    #         bucket[grp][tid] = []
    #         bucket[grp][tid].append(item)
    #         # print(key, item, tid)
    #         # input("@")
    
    # # Initialize all group in data[key]
    # now = st_time
    # for grp in data[key]:
    #     data[key][grp] = [0] * total_size

    # # To reasign data
    # for grp in bucket:
    #     for i in bucket[grp]:
            
    #         if len(bucket[grp][i]) > 1:
    #             bucket[grp][i] = reduce(lambda x, y: x + y, bucket[grp][i]) / len(bucket[grp][i])

    #         data[key][grp][i] = bucket[grp][i][0]
            
    # # To append 0 to missing data
    # for i, grp in enumerate(data[key]):
    #     mask_list[key][grp] = []
    #     miss_list[key][grp] = []
    #     miss_len_list[key][grp] = 0
    #     now = st_time
    #     for idx, item in enumerate(data[key][grp]):
            
    #         # If the data is missing, append 1 to mask_list
    #         if item == 0:
    #             w = get_week(now)
    #             mask_list[key][grp].append(1)
    #             miss_list[key][grp].append(get_date_string(now) )
    #             # density, flow, speed, week, timestamp, longitude, latitude
    #             data[key][grp][idx] = [0, 0, 0, w, now, 0, 0]
    #         else:
    #             mask_list[key][grp].append(0)
    #         now += 300

    #     print(key, grp, len(data[key][grp]), len(miss_list[key][grp]) )

    #     miss_len_list[key][grp] = ( len(miss_list[key][grp]) )
    # input("@")        

# find 30 consecutive (0,0,0) data and seperate data by each vd
# for vd in data:
#     for grp in data[vd]:
#         target_name = vd + "_" + grp
#         print(target_name)

#         # find 30 consecutive (0,0,0) data 
#         count = 0
#         for ptr, item in enumerate(data[vd][grp]):
            
#             if item[0] == 0 and item[1] == 0 and item[2] == 0:
                
#                 count += 1
#             else :
#                 if count >= time_period:
#                     for k in range(count):
#                         mask_list[vd][grp][ptr-k-1] = 1
#                 # print(item, count, mask_list[vd][grp][ptr])
#                 # input("!")
#                 count = 0
                    
#         print("finish finding 30 (0,0,0)s")
#         np.save( FIX_DATA_PATH + target_name, data[vd][grp])
#         np.save( MASK_DATA_PATH + target_name, mask_list[vd][grp])

process_ed = datetime.now()
print("finish... ", (process_ed-process_st) )
exit()

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

