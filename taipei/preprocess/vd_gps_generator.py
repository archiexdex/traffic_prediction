import os
import csv
import sys
import time
import datetime
import numpy as np

root_path = "/home/xdex/Documents/VD/"

file_name = "VD_GPS"

fd = open(root_path + file_name + ".csv")
csv_cursor = csv.reader(fd)

key = ""
x = y = 0

data = {}

# ['DEVICEID', 'DEVICEKIND', 'LOCATION', 'WGSX', 'WGSY']
#     0             1            2          3       4
for i, item in enumerate(csv_cursor):
    
    if i == 0:
        print(item)
        continue
    
    key = item[0]
    x = float(item[3])
    y = float(item[4])
    data[key] = [x, y]
    print(i)


print(len(data))    
np.save(file_name, data)
