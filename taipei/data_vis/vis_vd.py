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

DATA_PATH = '/home/xdex/Desktop/traffic_flow_detection/taipei/training_data/'
DATA_TRAIN_PATH = '/home/jay/Desktop/traffic_flow_detection/taipei/preprocess/'
FOLDER_PATH = '/home/jay/Desktop/traffic_flow_detection/taipei/data_vis/2YEAR_VP8GX40/'

DRAW_ONLINE_FLAG = False
# VD_NAMES = ['VMTG520', 'VMQGX40', 'VM7FI60', 'VLMG600', 'VLYGU40', 'VN5HV60', 'VN5HV61', 'VLRHT00']
# VD_NAMES = ['VP8GI60', 'VNNFY00', 'VP8GX40', 'VP8GX00', 'VP8GI20']
VD_NAMES = ['VMQGX40']
# VD_NAMES = ['VMTG520', 'VMQGX40', 'VLRHT00']
START_DAYS = [1]
DURATION = 600


def plot_duration(raw_data, vd_name, group_id, start_day, duration):
    """
    Params:
        raw_data:
        vd_name:
        group_id:
        start_day:
        duration:
    Return:
    """
    target_vd_data = np.array(raw_data[vd_name][group_id])[
        start_day * 12 * 24:(start_day + duration) * 12 * 24, 0:5]
    # print(target_vd_data.shape)

    # Add data
    start_week_day = target_vd_data[140, 3]
    start_date = datetime.datetime.fromtimestamp(
        target_vd_data[140, 4]).strftime("%Y-%m-%d")
    end_date = datetime.datetime.fromtimestamp(
        target_vd_data[-140, 4]).strftime("%Y-%m-%d")
    target_vd_density = target_vd_data[:, 0]
    target_vd_flow = target_vd_data[:, 1]
    target_vd_speed = target_vd_data[:, 2]
    target_vd_timestamp = target_vd_data[:, 4]
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
    layout = dict(title="VD_NAME: %s, GROUP_ID: %s, START_DATE: %s, END_DATE: %s, START_WEEK_DAY: %d" % (vd_name, group_id, start_date, end_date, start_week_day),
                  xaxis=dict(title='Time'),
                  yaxis=dict(title='Value'),
                  )

    fig = dict(data=data, layout=layout)
    if DRAW_ONLINE_FLAG:
        py.plot(fig, filename="VD_NAME: %s, GROUP_ID: %s, START_DATE: %s, END_DATE: %s, START_WEEK_DAY: %d" %
                (vd_name, group_id, start_date, end_date, start_week_day))
    else:
        plotly.offline.plot(fig, filename=FOLDER_PATH + "VD_NAME: %s, GROUP_ID: %s, START_DATE: %s, END_DATE: %s, START_WEEK_DAY: %d.html" %
                            (vd_name, group_id, start_date, end_date, start_week_day))


def main():
    # load raw data
    raw_filename = DATA_PATH + 'fix_raw_data.json'
    with codecs.open(raw_filename, 'r', 'utf-8') as f:
        raw_data = json.load(f)

    # plot
    for start_day in START_DAYS:
        for vd_name in VD_NAMES:
            for group_id in raw_data[vd_name]:
                plot_duration(raw_data, vd_name, group_id, start_day, DURATION)
                print("Finished:: VD_NAME: %s, GROUP_ID: %s, START_DAY: %d" %
                    (vd_name, group_id, start_day))


if __name__ == '__main__':
    if not os.path.exists(FOLDER_PATH):
        os.mkdir(FOLDER_PATH)
    main()
