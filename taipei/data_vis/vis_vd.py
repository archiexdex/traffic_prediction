"""

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import codecs
import os
import json
import matplotlib.pyplot as plt
import datetime
import plotly
plotly.__version__
plotly.tools.set_credentials_file(
    username='ChenChiehYu', api_key='xh9rsxFXY6DNF1qAfUyQ')
import plotly.plotly as py
import plotly.graph_objs as go

# DATA_PATH = '/home/xdex/Desktop/traffic_flow_detection/taipei/training_data/new_raw_data/vd_base/5/fix_data_group/'
DATA_PATH = '/home/xdex/Desktop/traffic_flow_detection/taipei/training_data/old_Taipei_data/vd_base/fix_data/'
OUTLIER_PATH = '/home/xdex/Desktop/traffic_flow_detection/taipei/training_data/old_Taipei_data/vd_base/mask_data/'
MASK_PATH = '/home/xdex/Desktop/traffic_flow_detection/taipei/training_data/old_Taipei_data/vd_base/mask_outlier/'
FOLDER_PATH = '/home/xdex/Desktop/traffic_flow_detection/taipei/data_vis/NEW_VDS/'


DRAW_ONLINE_FLAG = False

VD_NAMES = ["VQGFY00_0"]
START_DAYS = [1]
DURATION = 600


def plot_duration(vd_name, start_day, duration, is_perday=False):
    """
    Params:
        vd_name:
        start_day:
        duration:
    Return:
    """
    raw_filename = DATA_PATH + vd_name + '.npy'
    target_vd_data = np.load(raw_filename)[start_day * 12 * 24:(start_day + duration) * 12 * 24, 0:5]
    raw_filename = MASK_PATH + vd_name + ".npy"
    mask_vd_data = np.load(raw_filename)[start_day * 12 * 24:(start_day + duration) * 12 * 24]
    raw_filename = OUTLIER_PATH + vd_name + ".npy"
    outlier_vd_data = np.load(raw_filename)[start_day * 12 * 24:(start_day + duration) * 12 * 24]

    print(target_vd_data.shape)
    print(mask_vd_data.shape)
    print(outlier_vd_data.shape)

    # if mask or ourlier is 1, [d, f, s] shoud be [0, 0, 0]
    for idx in range(target_vd_data.shape[0]):
        if mask_vd_data[idx] == 1 or outlier_vd_data[idx] == 1:
            # print(idx)
            target_vd_data[idx, 1] = 0
            target_vd_data[idx, 2] = 0
            target_vd_data[idx, 3] = 0
    
    # Draw data for each day average from data with each 5 minutes
    if is_perday:
        target_vd_data_per_day = np.zeros([duration, 5])
        for idx, _ in enumerate(target_vd_data_per_day):
            target_vd_data_per_day[idx] = np.sum(target_vd_data[idx * 12 * 24: (idx+1) * 12 * 24, 0:5], axis=0) / (12 * 24)
        
        start_week_day = target_vd_data_per_day[0, 4]
        start_date = datetime.datetime.fromtimestamp(
            target_vd_data_per_day[0, 0]).strftime("%Y-%m-%d")
        end_date = datetime.datetime.fromtimestamp(
            target_vd_data_per_day[duration-1, 0]).strftime("%Y-%m-%d")
        target_vd_density = target_vd_data_per_day[:, 1]
        target_vd_flow = target_vd_data_per_day[:, 2]
        target_vd_speed = target_vd_data_per_day[:, 3]
        target_vd_timestamp = target_vd_data_per_day[:, 0]
    
    # Draw data for each 5 minutes
    else:
        start_week_day = target_vd_data[140, 4]
        start_date = datetime.datetime.fromtimestamp(
            target_vd_data[140, 0]).strftime("%Y-%m-%d")
        end_date = datetime.datetime.fromtimestamp(
            target_vd_data[-140, 0]).strftime("%Y-%m-%d")
        target_vd_density = target_vd_data[:, 1]
        target_vd_flow = target_vd_data[:, 2]
        target_vd_speed = target_vd_data[:, 3]
        target_vd_timestamp = target_vd_data[:, 0]
    
    time_list = []
    for _, v in enumerate(target_vd_timestamp):
        time_list.append(datetime.datetime.fromtimestamp(
            v).strftime("%Y-%m-%d %H:%M:%S"))

    # Create and style traces
    trace_density = go.Scatter(
        x=time_list,
        y=target_vd_density,
        name='Density',
        line=dict(
            color=('rgb(12, 205, 24)'),
            width=3)
    )
    trace_flow = go.Scatter(
        x=time_list,
        y=target_vd_flow,
        name='Flow',
        line=dict(
            color=('rgb(24, 12, 205)'),
            width=3)
    )
    trace_speed = go.Scatter(
        x=time_list,
        y=target_vd_speed,
        name='Speed',
        line=dict(
            color=('rgb(205, 12, 24)'),
            width=3)
    )
    data = [trace_density, trace_flow, trace_speed]

    # Edit the layout
    layout = dict(title="VD_NAME: %s, START_DATE: %s, END_DATE: %s, START_WEEK_DAY: %d" % (vd_name, start_date, end_date, start_week_day),
                  xaxis=dict(title='Time'),
                  yaxis=dict(title='Value'),
                  )

    fig = dict(data=data, layout=layout)
    if DRAW_ONLINE_FLAG:
        py.plot(fig, filename="VD_NAME: %s, START_DATE: %s, END_DATE: %s, START_WEEK_DAY: %d" %
                (vd_name, start_date, end_date, start_week_day))
    else:
        plotly.offline.plot(fig, filename=FOLDER_PATH + "VD_NAME: %s, START_DATE: %s, END_DATE: %s, START_WEEK_DAY: %d.html" %
                            (vd_name, start_date, end_date, start_week_day))


def main():
    # plot
    for start_day in START_DAYS:
        for vd_name in VD_NAMES:
            plot_duration(vd_name, start_day, DURATION, is_perday=True)
            # plot_duration(vd_name, start_day, DURATION, is_perday=False)
            print("Finished:: VD_NAME: %s, START_DAY: %d" %
                  (vd_name, start_day))


if __name__ == '__main__':
    if not os.path.exists(FOLDER_PATH):
        os.mkdir(FOLDER_PATH)
    main()
