import os
import csv
import json


file_path = "/home/xdex/Documents/Taipei_xml_data/VD_GPS.csv"
save_path = "/home/xdex/Desktop/traffic_flow_detection/taipei/training_data/old_Taipei_data/vd_base/"

file = open(file_path)
csv_cursor = csv.reader(file)

data = {}

for i, item in enumerate(csv_cursor):
    # print(item)
    # input("@")
    # First row
    if i == 0:
        # print(item)
        # input("@@")
        continue
    
    vd_id    = item[0]
    is_plane = item[9]
    lat      = float(item[19])
    lon      = float(item[20])
    if lat < 20 or lon < 120:
        continue
    
    if vd_id not in data:
        data[vd_id] = {}
        if is_plane == "高架":
            data[vd_id]["is_plane"] = False
        else:
            data[vd_id]["is_plane"] = True
        
        data[vd_id]["lat"] = lat
        data[vd_id]["lon"] = lon

with open(os.path.join(save_path, "vd_info.json"), mode="w" ) as fp:
    json.dump(data, fp)