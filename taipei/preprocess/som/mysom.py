# Python 2
import os
import json 
import math
import numpy as np
from sklearn import preprocessing
from model_som import SOM
import matplotlib.pyplot as plt

root_path = "/home/xdex/Desktop/traffic_flow_detection/taipei/training_data/"
m = 30
n = 30

print("Reading vd_list...")
vd_list = []
for root, dirs, files in os.walk(root_path + "new_raw_data/vd_base/5/fix_data/"):
    tmp = {}
    for fp in files:
        tmp[fp[:7]] = 0
    for key in tmp:
        vd_list.append(key)
    break

print("Reading VD_GPS...")
vd_gps = np.load(root_path + "new_raw_data/VD_GPS.npy").item()

print("Making input data...")
data = []
label_data = []
for vd in vd_list:
    if vd not in vd_gps:
        continue
    data.append(vd_gps[vd])
    label_data.append(vd)

# Normalization
data = preprocessing.MinMaxScaler().fit_transform(data)
data = np.array(data)

grid = []
try:
    grid = np.load( "som_grid.npy")
except:
    pass

if grid == []:
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
ret_dict = {}
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
    ret_dict[label_data[idx]] = ptr
    check_map[ptr[0]][ptr[1]] = 1  
    if label_data[idx] == "VIKPW61":
        
        print("rec_dist [vd]:", label_data[idx], ret_dict[label_data[idx]], ptr)
    # qq = raw_input("!")

print(len(data))
print(ret_dict)

with open(root_path + "som_list.json", "w") as fp:
    json.dump(ret_dict, fp)

