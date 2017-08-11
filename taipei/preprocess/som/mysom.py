# Python 2
from sklearn import preprocessing
from model_som import SOM
import numpy as np
import matplotlib.pyplot as plt
import json 
import math


root_path = "/home/xdex/Desktop/traffic_flow_detection/taipei/training_data/"
m = 10
n = 10

print("Reading vd_list...")
vd_list = []
with open(root_path + "selected_vd.json") as fp:
    tmp = json.load(fp)
    vd_list = tmp["label"]["1"]

print("Reading VD_GPS...")
vd_gps = np.load(root_path + "VD_GPS.npy").item()

print("Making input data...")
data = []
label_list = []
with open(root_path + "fix_raw_data.json") as fp:
    tmp = json.load(fp)
    for key in vd_list:
        for grp in tmp[key]:
            data.append(vd_gps[key])   
            label_list.append(key)

# Normalization
data = preprocessing.MinMaxScaler().fit_transform(data)
data = np.array(data)

# Initialize SOM and training
som = SOM(m,n,2)
som.train(data)

#Get output grid
image_grid = som.get_centroids()

#Map colours to their closest neurons
mapped = som.map_vects(data)

grid = np.array(image_grid)

np.save("som_grid", grid)

# Find which vd in which grid
check_map = np.zeros([m,n])
ret_map = np.zeros([m,n]).tolist()
for idx, v in enumerate(data):
    mi = 123456789
    ptr = [-1, -1]
    for i, row in enumerate(grid):
        for j, item in enumerate(row):
            # calculate the distance between two node
            dist = np.linalg.norm(v-item)
            
            if mi > dist and check_map[i][j] == 0:
                mi = dist
                ptr = [i, j]
    ret_map[ptr[0]][ptr[1]] = label_list[idx]
    check_map[ptr[0]][ptr[1]] = 1  

print(ret_map)
np.save("som_map", ret_map)

