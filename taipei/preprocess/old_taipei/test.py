import numpy as np
import codecs
import json
import os
import csv
import scipy.stats as stats
import datetime
from sklearn import preprocessing

root_path = "/home/xdex/Desktop/traffic_flow_detection/taipei/training_data/old_Taipei_data/vd_base/"
data = np.load(os.path.join(root_path, "test_data_train_100_train_100.npy") )[-1,:,:,0:1+3]
x = np.arange(-2.,-2+6)

now = datetime.datetime.fromtimestamp(data[-1][2][0])

with open( os.path.join(root_path, "vd_grp_lane.json") ) as fp:
    vd_grp_lane = json.load(fp)

# with open( os.path.join(root_path, "target_vd_list.json") ) as fp:
#     target_vd_list = json.load(fp)
with open( "target_vd_list.json", mode='r') as fp:
    target_vd_list = json.load(fp)

with open( os.path.join(root_path, "vd_info.json") ) as fp:
    vd_info = json.load(fp)

errors = np.zeros( (data.shape[0], 3) ) 
# Calculate regression for error in each VD
# errors = np.zeros( (data.shape[0], 3) ) 
# for idx, vd in enumerate(target_vd_list["total"]):
#     for jdx in range(3):
#         slope, intercept, r_value, p_value, std_err = stats.linregress(x, data[idx, :, jdx])
#         errors[idx, jdx] = std_err

# # Normalization
# for idx in range(3):
#     errors[:, idx] = [ ptr / max(errors[:, idx]) for ptr in errors[:, idx] ]
    # errors[:, idx] = errors[:, idx] / np.linalg.norm(errors[:, idx])
        
vd_gps = np.load(os.path.join(root_path, "VD_GPS.npy") ).item()

output =[[["VD_Name", "latitude", "longitude", "time", "is_Plane", "lane_amount", "loss"] + [str(datetime.datetime.fromtimestamp(data[-1][i][0])) for i in range(data.shape[1])]],
         [["VD_Name", "latitude", "longitude", "time", "is_Plane", "lane_amount", "loss"] + [str(datetime.datetime.fromtimestamp(data[-1][i][0])) for i in range(data.shape[1])]],
         [["VD_Name", "latitude", "longitude", "time", "is_Plane", "lane_amount", "loss"] + [str(datetime.datetime.fromtimestamp(data[-1][i][0])) for i in range(data.shape[1])]]]

# Remove time
data = data[:,:,1:1+3]

for idx, vd in enumerate(target_vd_list["total"]):
    # density flow speed
    # print(vd_info[vd]["is_plane"])
    the_vd, the_group= vd.split("_")
    # print(len(vd_grp_lane[the_vd][the_group]) )
    #tmp = [vd, vd_gps[the_vd][0], float(vd_gps[the_vd][1]) + float(the_group) * 0.00001]
    #output.append(tmp)
    #try:
    for jdx in range(3):
        tmp = [vd, vd_gps[the_vd][0], float(vd_gps[the_vd][1]) + float(the_group) * 0.00001, str(now), vd_info[the_vd]["is_plane"], len(vd_grp_lane[the_vd][the_group])]
        for kdx in range(data.shape[1]):
            tmp.append(data[idx, kdx, jdx])
        tmp.append(errors[idx, jdx])
        output[jdx].append(tmp)
    #except:
    #print(vd)
        

#with open("vd_gps.csv", "w") as fp:
#    csv_writer = csv.writer(fp)
#    csv_writer.writerows(output)

with open("density.csv", "w") as fp:
    csv_writer = csv.writer(fp)
    csv_writer.writerows(output[0])
with open("flow.csv", "w") as fp:
    csv_writer = csv.writer(fp)
    csv_writer.writerows(output[1])
with open("speed.csv", "w") as fp:
    csv_writer = csv.writer(fp)
    csv_writer.writerows(output[2])
