import numpy as np
import os
import json
from datetime import datetime

root_path = "../"

vd_gps = np.load(root_path+"VD_GPS.npy").item()
data = np.load(root_path+"input_data.npy").item()

vd_list = []
with open(root_path+"vd_list") as vd:
    for i in vd:
        key = i.strip()
        vd_list.append(key)

st_time = 102345678900
ed_time = -102345678900
for key in vd_list:

    #print(key, len(data[key]))
    for i in data[key]:
        
        if st_time > i[4]:
            st_time = i[4]
        if ed_time < i[4]:
            ed_time = i[4]


print(st_time, ed_time)
def get_week(timestamp):
    return (datetime.fromtimestamp(timestamp).timetuple()[6] + 1 ) % 7

def get_date_string(timestamp):
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

print(datetime.fromtimestamp(st_time).timetuple())
print(datetime.fromtimestamp(ed_time)) 

#exit()
miss_list = {}
miss_len_list = {}
mask_list = {}
bucket = {}
total_size = int(ed_time - st_time) // 300 + 1

for key in vd_list:
    flg = 0
    miss_list[key] = []
    miss_len_list[key] = []
    mask_list[key] = []
    bucket = {}

    # To get redundant data
    for i, item in enumerate(data[key]):
        id = int(item[4]-st_time) // 300

        bucket[id] = []
        bucket[id].append(item)
    
    data[key] = [0] * total_size
    # To reasign data
    for i in bucket:
        # print(i, bucket[i])
        if len(bucket[i]) > 1:
            bucket[i] = reduce(lambda x, y: x + y, bucket[i]) / len(bucket[i])

        # print(i, bucket[i][0])
        data[key][i] = bucket[i][0]

    now = st_time
    for i, item in enumerate(data[key]):
        if item == 0:
            w = get_week(now)
            mask_list[key].append(1)
            miss_list[key].append(get_date_string(now) )
            data[key][i] = [0, 0, 0, w, now]
        else:
            mask_list[key].append(0)
        now += 300

    print(key, len(data[key]))

    print(key, len(miss_list[key]) )
    miss_len_list[key].append( len(miss_list[key]) )
    # input("@")        

# with open('miss_len_data.json', 'w') as fp:
#     json.dump(miss_len_list, fp)

# with open('miss_data.json', 'w') as fp:
#     json.dump(miss_list, fp)

# with open('fix_raw_data.json', 'w') as fp:
#     json.dump(data, fp)

with open(root_path+'mask_list.json', 'w') as fp:
    json.dump(mask_list, fp)

# np.save("fix_input_data", data)
