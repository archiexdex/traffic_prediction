import numpy as np
import json 
import time
from datetime  import datetime

root_path = "/home/xdex/Desktop/traffic_flow_detection/taipei/training_data/"
st_time = time.mktime( datetime.strptime("2015-01-01 00:05:00", "%Y-%m-%d %H:%M:%S").timetuple() )
raw_data = {}

def get_week(timestamp):
    return (datetime.fromtimestamp(timestamp).timetuple()[6] + 1 ) % 7

print("Opening fix_raw_data")
with open(root_path + "fix_raw_data.json") as fp:
    raw_data = json.load(fp)


print("Starting to fix time")
for key in raw_data:
    for grp in raw_data[key]:
        for item in raw_data[key][grp]:
            hour = datetime.fromtimestamp(item[4]).timetuple()[3]
            minute = datetime.fromtimestamp(item[4]).timetuple()[4]
            item[4] = hour*60+minute

print("Saving fix_raw_data")
with open(root_path + "fix_raw_data_time1.json", "w") as fp:
    json.dump(raw_data, fp)