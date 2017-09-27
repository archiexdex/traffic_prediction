import numpy as np
import xml.etree.ElementTree as ET
import datetime
import time
import urllib.request
import os
import sys

ROOT_PATH  = "http://140.113.210.14:6006/DataBase/Taipei_current_data/xml/"
SAVE_PATH  = "/home/xdex/Desktop/traffic_flow_detection/taipei/training_data/current_data/lane_base/"

START_TIME = datetime.datetime(2017,9,21,16,00)
END_TIME   = datetime.datetime(2017,9,22,16,10)


def get_next_time(_minute):
    return START_TIME + datetime.timedelta(minutes=_minute)

# 0 is Sunday
def get_week(timestamp):
    return (datetime.datetime.fromtimestamp(timestamp).timetuple()[6] + 1 ) % 7

def main():
    
    now = START_TIME
    count = -5
    data = {}
    # enumerate time from start_time to end_time
    while now != END_TIME:
        count += 5
        now = get_next_time(count)
        file_path = os.path.join( ROOT_PATH , (now.strftime("%Y%m%d_%H%M") + ".xml") )
        
        # try to get data from server
        try:
            xml_data = urllib.request.urlopen(file_path)
        except: 
            print(file_path, " doesn't exists!!")
            continue
        
        # parse xml data
        tree = ET.ElementTree(file=xml_data)

        VD_ID = ""
        DENSITY = 0.0
        FLOW = 0.0
        SPEED = 0.0
        TIMESTAMP = time.mktime( now.timetuple() )
        WEEK = get_week(time.mktime( now.timetuple()) ) 
        LANEID = 0

        # enumerate xml tree
        for elem in tree.iter():
            # print(elem.tag, elem.attrib, elem.text, data)
            # input("@")
            if elem.tag == "DeviceID":
                if VD_ID != elem.text:
                    VD_ID = elem.text
                    if VD_ID not in data:
                        data[VD_ID] = {}
                pass
            
            if elem.tag == "LaneNO":
                LANEID = int(elem.text)
                if LANEID not in data[VD_ID]:
                    data[VD_ID][LANEID] = []
                pass
            
            if elem.tag == "Volume":
                FLOW = float(elem.text)
                pass

            if elem.tag == "AvgSpeed":
                SPEED = float(elem.text)
                pass

            if elem.tag == "AvgOccupancy":
                DENSITY = float(elem.text)
                data[VD_ID][LANEID].append( [TIMESTAMP, DENSITY, FLOW, SPEED, WEEK] )
                pass
    
    # save data by each vd
    for vd in data:
        for lane in data[vd]:
            path = os.path.join(SAVE_PATH, vd+"_"+str(lane) )
            np.save(path, data[vd][lane])
    pass


if __name__ == "__main__":
    main()
