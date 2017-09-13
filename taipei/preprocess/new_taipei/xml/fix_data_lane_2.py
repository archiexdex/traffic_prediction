from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import datetime
import time
import argparse
import json
import threading
import numpy as np

"""
    Variables
"""
parser = argparse.ArgumentParser(
    description='process aid to fix each data for each lane.')
parser.add_argument("--mode", action='store', dest='mode', type=int,
                    help="you can choose mode=1 or mode=5 for different time period. Default is 5")
parser.add_argument("--time_padding", action='store', dest='time_padding', type=int,
                    help="set the time period that how long (0,0,0) can exist. Default is 30.")
parser.add_argument("--is_allow_offset", action='store', dest='is_allow_offset', type=int,
                    help="set 0 means process won't check how long (0,0,0) exist. Default is 1.")
parser.add_argument("--long_period", action='store', dest='long_period', type=int,
                    help="1 means time = mode * 1 e.g. long_period = 1 and mode = 5 means if there are more than 5 minutes data be [0, 0, 0], we will mask it")
parser.add_argument("--is_append_mask", action='store', dest='is_append_mask', type=int,
                    help="is append mask channel in train data")
args = parser.parse_args()
# choose 1 or 5 to fix different mode data
mode = 5
time_padding = 30
is_allow_offset = 1
is_append_mask = 1
# 1 means time = mode * 1 e.g. long_period = 1 and mode = 5 means if there
# are more than 5 minutes data be [0, 0, 0], we will mask it
long_period = 6

if args.mode != None:
    mode = args.mode
if args.time_padding != None:
    time_padding = args.time_padding
if args.is_allow_offset != None:
    is_allow_offset = args.is_allow_offset
if args.long_period != None:
    long_period = args.long_period
if args.is_append_mask != None:
    is_append_mask = args.is_append_mask

# 2015/12/01 00:00:00 ~ 2017/07/31 23:55:00
start_time = time.mktime(datetime.datetime.strptime(
    "2015-12-01 00:00:00", "%Y-%m-%d %H:%M:%S").timetuple())
end_time = time.mktime(datetime.datetime.strptime(
    "2017-08-01 00:00:00", "%Y-%m-%d %H:%M:%S").timetuple())
data_size = int((end_time - start_time) / 300)

st_time = datetime.datetime.now()

root_path = "/home/xdex/Desktop/traffic_flow_detection/taipei/training_data/new_raw_data/vd_base/"
data_path = ""
mask_path = ""
save_path = ""
vd_name = ""
if mode == 1:
    data_path = root_path + "1/data/"
    mask_path = root_path + "1/mask/"
    save_path = root_path + "1/fix_data/"
elif mode == 5:
    data_path = root_path + "5/data_lane/"
    mask_path = root_path + "5/mask_lane/"
    save_path = root_path + "5/fix_data_lane/"

"""
    Function
"""
# 0 is Sunday


def get_week(timestamp):
    # Because the timetuple return 0 is Monday, so I add one to change 1 to
    # Monday
    return (datetime.datetime.fromtimestamp(timestamp).timetuple()[6] + 1) % 7


def get_hour(timestamp):
    # timetuple : year month day hour minute second week ...
    return datetime.datetime.fromtimestamp(timestamp).timetuple()[3]

# Save mask and fix_data


def check_data(path):
    data = np.load(path)
    fix_data = [0] * data_size
    mask_list = [0] * data_size
    now = start_time
    ptr = 0
    i = 0
    count = 0

    while True:
        # item : [time, density, flow, speed, week]
        item = []
        if i < data.shape[0]:
            item = data[i].tolist()

        if now >= end_time:
            break

        # It may occur when data has duplicate time or begging time is lower
        # than start_time
        if i < data.shape[0] and abs(item[0] - now) <= time_padding:
            i += 1
            continue
        # It may occur when data has missing data
        elif i >= data.shape[0] or abs(item[0] - now) <= time_padding:
            if is_append_mask == 0:
                fix_data[ptr] = [now, 0, 0, 0, get_week(now)]
            elif is_append_mask == 1 :
                fix_data[ptr] = [now, 0, 0, 0, get_week(now), 0, now]
            mask_list[ptr] = 1
            ptr += 1

        else:
            # To check if [density, flow, speed] is [0, 0, 0] and continue in
            # long_period, then they are missing data
            if is_allow_offset == 1:
                if item[1:1 + 3] == [0, 0, 0] and (8 <= get_hour(item[0]) and get_hour(item[0]) <= 22):
                    count += 1
                else:
                    if count > long_period:
                        for k in range(count):
                            mask_list[ptr - k - 1] = 1
                        count = 0

            if is_append_mask == 1:
                item.append(0)
                item.append(now)
            fix_data[ptr] = item
            ptr += 1
            i += 1

        if mode == 1:
            now += 60
        elif mode == 5:
            now += 300
    # print("Saving fix_data...")
    np.save(save_path + vd_name, fix_data)
    # print("Saving mask_data...")
    np.save(mask_path + vd_name, mask_list)


"""
    Main
"""
for root, dirs, files in os.walk(data_path):
    for file in files:
        path = os.path.join(data_path, file)
        print("Fixing VD: "+file)
        vd_name = file[:9]
        check_data(path)
    break

# file_list = []
# for root, dirs, files in os.walk(data_path):
#     file_list = files
#     break


# class MyThread(threading.Thread):
#     def __init__(self, thread_id, thread_name, file_list):
#         threading.Thread.__init__(self)
#         self.thread_id = thread_id
#         self.thread_name = thread_name
#         self.file_list = file_list

#     def run(self):
#         for file in self.file_list:
#             path = os.path.join(data_path, file)
#             print("Fixing VD: " + file)
#             vd_name = file[:9]
#             check_data(path)


# file_size = len(file_list)
# # Initialize new threads
# thread1 = MyThread(
#     1, "Thread-1", file_list[file_size // 8 * 0:file_size // 8 * 1])
# thread2 = MyThread(
#     2, "Thread-2", file_list[file_size // 8 * 1:file_size // 8 * 2])
# thread3 = MyThread(
#     3, "Thread-3", file_list[file_size // 8 * 2:file_size // 8 * 3])
# thread4 = MyThread(
#     4, "Thread-4", file_list[file_size // 8 * 3:file_size // 8 * 4])

# thread5 = MyThread(
#     5, "Thread-5", file_list[file_size // 8 * 4:file_size // 8 * 5])
# thread6 = MyThread(
#     6, "Thread-6", file_list[file_size // 8 * 5:file_size // 8 * 6])
# thread7 = MyThread(
#     7, "Thread-7", file_list[file_size // 8 * 6:file_size // 8 * 7])
# thread8 = MyThread(8, "Thread-8", file_list[file_size // 8 * 7:])

# thread_list = [thread1, thread2, thread3,
#                thread4, thread5, thread6, thread7, thread8]
# # Start threads
# for i in thread_list:
#     i.start()

# # Join threads
# for i in thread_list:
#     i.join()


ed_time = datetime.datetime.now()
print("Finishing...", ed_time - st_time)
