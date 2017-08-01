import json
import numpy as np
import os


root_path = "../"

VD_GPS = np.load(root_path + "VD_GPS.npy").item()

vd_list = []
with open(root_path+"vd_list") as vd:
    for i in vd:
        key = i.strip()
        vd_list.append(key)


vd_gps_list = []

for vd in vd_list:
    gps = VD_GPS[vd]
    vd_gps_list.append( (vd, gps) )


x_base = sorted(vd_gps_list, key=lambda x: x[1][0] )
y_base = sorted(vd_gps_list, key=lambda x: x[1][1] )

ret = {}
ret['x_base'] = list(map(lambda x: x[0], x_base) )
ret['y_base'] = list(map(lambda x: x[0], y_base) )


with open(root_path+'reduce_dimension.json', 'w') as fp:
    json.dump(ret, fp)