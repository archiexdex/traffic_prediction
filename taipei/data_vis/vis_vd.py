"""

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import codecs
import json
import matplotlib.pyplot as plt
import datetime
import plotly
plotly.__version__
plotly.tools.set_credentials_file(
    username='ChenChiehYu', api_key='xh9rsxFXY6DNF1qAfUyQ')
import plotly.plotly as py
import plotly.graph_objs as go

DATA_PATH = '/home/xdex/Desktop/traffic_flow_detection/taipei/'
DATA_TRAIN_PATH = '/home/jay/Desktop/traffic_flow_detection/taipei/preprocess/'

VD_NAME = "VP8GX40"
GROUP_ID = 0
DAY = 97

def plot_one_day(vd_name, group_id, day):
    """
    Params:
        vd_name:
        group_id:
        day:
    Return:
    """
    # load raw data
    raw_filename = DATA_PATH + 'fix_raw_data.json'
    with codecs.open(raw_filename, 'r', 'utf-8') as f:
        raw_data = json.load(f)
    target_vd_data = np.array(list(raw_data[vd_name].values())[group_id])[
        day * 12 * 24:(day + 1) * 12 * 24, 0:5]
    print(target_vd_data.shape)

    # Add data
    week_day = target_vd_data[140, 3]
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
            width=4)
    )
    trace_flow = go.Scatter(
        x=time_list,
        y=target_vd_flow,
        name='Flow',
        line=dict(
            color=('rgb(24, 12, 205)'),
            width=4)
    )
    trace_speed = go.Scatter(
        x=time_list,
        y=target_vd_speed,
        name='Speed',
        line=dict(
            color=('rgb(205, 12, 24)'),
            width=4)
    )
    data = [trace_density, trace_flow, trace_speed]

    # Edit the layout
    layout = dict(title="VD_NAME: %s, GROUP_ID: %d, DAY: %d, WEEK: %d" % (vd_name, group_id, day, week_day),
                  xaxis=dict(title='Time'),
                  yaxis=dict(title='Value'),
                  )

    fig = dict(data=data, layout=layout)
    py.plot(fig, filename="VD_NAME: %s, GROUP_ID: %d, DAY: %d, WEEK: %d" %
            (vd_name, group_id, day, week_day))


def main():
    plot_one_day(VD_NAME, GROUP_ID, DAY)

if __name__ == '__main__':
    main()
