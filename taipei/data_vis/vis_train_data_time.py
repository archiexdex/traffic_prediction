"""
To select the better area for training, we visaulize the rate of missing data and outliers by heatmap.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import numpy as np
import plotly
import datetime
import json
import codecs
plotly.__version__
plotly.tools.set_credentials_file(
    username='ChenChiehYu', api_key='xh9rsxFXY6DNF1qAfUyQ')
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.graph_objs import *
import pandas as pd

is_old = True

# PATH
DATA_PATH = '/home/xdex/Desktop/traffic_flow_detection/taipei/training_data/new_raw_data/vd_base/'
if is_old:
    DATA_PATH   = "/home/xdex/Desktop/traffic_flow_detection/taipei/training_data/old_Taipei_data/vd_base/"

FOLDER_PATH = 'all_vds_alltime_new/'
if is_old:
    FOLDER_PATH = 'train_data_time/'

TRAIN_VD_LIST = os.path.join(DATA_PATH, 'train_vd_list.json')
VD_TIME_LIST = os.path.join(DATA_PATH, 'train_vd_time_list.npy')

def get_date(day):
    if is_old:
        return (datetime.date(2015, 1, 1) + datetime.timedelta(days=day)).strftime("%Y%m%d")
    return (datetime.date(2015, 12, 1) + datetime.timedelta(days=day)).strftime("%Y%m%d")

def draw_discrete_heatmap(data):
    """draw heatmap by the rate of each type of statistics, including mssing data rate, outliers rate, both above attribute rate

    Params
    ------
    vd_name_list : 
    data : 
    day_idx : int

    Return
    ------
    nope

    Raises
    ------
    nope
    """
    time_range = [x for x in range(1440)]
    time_list = []
    for time in time_range:
        hours = time // 60
        mins = time % 60
        time_list.append('%d點%d分' % (hours, mins))

    # prepare vis
    # Create a trace
    trace = go.Scatter(
        x = time_list,
        y = data
    )

    data = [trace]
    
    # Edit the layout
    layout = dict(
                  xaxis=dict(title='Time'),
                  )

    fig = dict(data=data, layout=layout)
    # draw
    plotly.offline.plot(fig, filename=FOLDER_PATH +
                        "vd_time_list.html", auto_open=False)


def main():
    # read vd name list
    with codecs.open(TRAIN_VD_LIST, 'r', 'utf-8') as f:
        train_vd_list = json.load(f)
    vd_name_list = np.array(train_vd_list['train'])

    # read data (missing mask)
    all_data = np.load(VD_TIME_LIST)

    # draw
    
    draw_discrete_heatmap(all_data)

if __name__ == '__main__':
    if not os.path.exists(FOLDER_PATH):
        os.mkdir(FOLDER_PATH)
    main()
