import numpy as np
import os
import json
import types

file_path = "/home/xdex/Desktop/traffic_flow_detection/taipei/training_data/"

file_name_list  =  ["20150101000000_20150112000000",
                    "20150112000000_20150212000000",
                    "20150212000000_20150312000000",
                    "20150312000000_20150412000000",
                    "20150412000000_20150512000000",
                    "20150512000000_20150612000000",
                    "20150612000000_20150712000000",
                    "20150712000000_20150812000000",
                    "20150812000000_20150912000000",
                    "20150912000000_20151012000000",
                    "20151012000000_20151112000000",
                    "20151112000000_20151212000000",
                    "20151212000000_20160112000000",

                    "20160112000000_20160212000000",
                    "20160212000000_20160312000000",
                    "20160312000000_20160412000000",
                    "20160412000000_20160512000000",
                    "20160512000000_20160612000000",
                    "20160612000000_20160712000000",
                    "20160713000000_20160813000000",
                    "20160813000000_20160913000000",
                    "20160913000000_20161013000000",
                    "20161013000000_20161113000000",
                    "20161113000000_20161213000000",
                    "20161213000000_20170113000000",

                    "20170113000000_20170213000000",
                    "20170213000000_20170313000000",
                    ]


data = {}

vd_list = []
with open(file_path + "selected_vd.json") as file:
    tmp = json.load(file)
    vd_list = tmp["train"]
    # for i in tmp["train"]:
    #     vd_list.append(i)

ma = -10123456789
for file_name in file_name_list:
    tmp = np.load(file_path + "raw_data/" +file_name + ".npy").item()
    ma = max(len(tmp), ma)
    
    print("reading ", file_name)
    if len(data) == 0:
        for vd in vd_list:
            data[vd] = tmp[vd]
        
    else:
        for vd in vd_list:
            for vd_grp in tmp[vd]:
                
                if vd_grp not in data[vd]:
                    data[vd][vd_grp] = tmp[vd][vd_grp]
                    # print(tmp[vd][vd_grp])
                else:
                    for item in tmp[vd][vd_grp]:
                        data[vd][vd_grp].append(item)
                
    
print("max:", ma)

with open('raw_data.json', 'w') as fp:
    json.dump(data, fp)
# np.save("raw_data", data)

