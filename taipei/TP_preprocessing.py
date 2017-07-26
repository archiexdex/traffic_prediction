
import os
import csv
import sys
import time
import datetime
import numpy as np

root_path = "/home/xdex/Documents/VD/201501-201703/"

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
for file_name in file_name_list:
    file = open(root_path + file_name + ".csv")
    csv_cursor = csv.reader(file)


    def the_date(the_time):
        # return (datetime.date(2015, 1, 1) + datetime.timedelta(microseconds=day-1)).strftime("%Y-%m-%d %H:%M:%S")
        return datetime.datetime.fromtimestamp(the_time)

    def check_time(tt):
        # tm_year, tm_mon, tm_mday, tm_hour, tm_min, tm_sec, tm_wday, tm_yday, tm_isdst
        return tt[4] % 5 == 0 and tt[5] == 0

    # {key, [[density, flow, speed, week, time]]}
    data = {}

    lane_order = 0
    density = 0
    flow = 0
    speed = 0
    timestamp = 0
    week = 4
    count = 0
    flg = 0

    key_old = key_now = ""

    #      0           1            2            3           4           5            6              7             8           9            10          11       12       13
    # ['DEVICEID', 'LANEORDER', 'BIGVOLUME', 'BIGSPEED', 'CARVOLUME', 'CARSPEED', 'MOTORVOLUME', 'MOTORSPEED', 'AVGSPEED', 'LANEOCCUPY', 'DATETIME2', 'RATE', 'AVGINT', 'LGID']
    for i, item in enumerate(csv_cursor):

        if i == 0:
            print(item)
            input("@@")
            continue
        
        key_now = item[0]
        if key_old == key_now:
            density += float(item[9])
            flow    += float(item[2]) + float(item[4])
            speed   += float(item[8])
            count   += 1
            
        else:
            # Calculate last VD data
            if count > 0 :
                # print(density, flow, speed, week, timestamp)
                density /= count
                speed /= count 
                if key_old in data:
                    data[key_old].append([density, flow, speed, week, timestamp])
                else:
                    data[key_old] = [[density, flow, speed, week, timestamp]]
                density = flow = speed = timestamp = week = count = 0

            # Initial for next VD data
            key_old     = key_now
            tt = datetime.datetime.strptime(item[10], "%Y-%m-%d %H:%M:%S").timetuple()
            if check_time(tt) == False:
                flg += 1
                continue
            flg = 0
            density = float(item[9])
            flow    = float(item[2]) + float(item[4])
            speed   = float(item[8])

            timestamp  = time.mktime( tt ) 
            week = (tt[6] + 1) % 7
            count   = 1
        
        print(item)
        input("!!")
    np.save(file_name, data)

    print(len(data))
