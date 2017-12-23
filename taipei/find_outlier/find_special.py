import json
import numpy as np
import os

DATA_PATH = '/home/xdex/Desktop/traffic_flow_detection/taipei/training_data/old_Taipei_data/vd_base/'

var_list = {}
with open("tmp.json") as fp:
    var_list = json.load(fp)

with open( DATA_PATH + "target_vd_list.json") as fp:
    vd_list = json.load(fp)

data = []
for vd in var_list:
    data.append([vd, var_list[vd]])

data = sorted(data, reverse=True, key=lambda x: x[1])
for idx, item in enumerate(data):
    if idx > 50:
         break
    print(item)