"""
To select the better area for training, we visaulize the rate of missing data and outliers by heatmap.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import numpy as np
import plotly
import json
import codecs
plotly.__version__
plotly.tools.set_credentials_file(
    username='ChenChiehYu', api_key='xh9rsxFXY6DNF1qAfUyQ')
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.graph_objs import *
import pandas as pd

# PATH
DATA_PATH = '/home/xdex/Desktop/traffic_flow_detection/taipei/training_data/new_raw_data/vd_base/'
FOLDER_PATH = 'all_vds_30/'

TRAIN_VD_LIST = os.path.join(DATA_PATH, 'train_vd_list.json')
VD_TIME_MASK = os.path.join(DATA_PATH, 'vd_time_mask_30.npy')

def draw_discrete_heatmap(vd_name_list, data, day_idx):
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
    target_data = data[day_idx]
    time_range = [x * 5 for x in range(288)]
    time_list = []
    for time in time_range:
        hours = time // 60
        mins = time % 60
        time_list.append('%d點%d分' % (hours, mins))

    # prepare vis
    missing_trace = go.Heatmap(
        x=time_list, y=vd_name_list, z=target_data, name='missing rate')
    data = [missing_trace]

    # Edit the layout
    layout = dict(title="statistics of VD missings on day %d" % (day_idx),
                  xaxis=dict(title='Time'),
                  yaxis=dict(title='VD Name'),
                  )

    fig = dict(data=data, layout=layout)
    # draw
    plotly.offline.plot(fig, filename=FOLDER_PATH +
                        "statistics of VD missings on day %d.html" % (day_idx))


def main():
    # read vd name list
    with codecs.open(TRAIN_VD_LIST, 'r', 'utf-8') as f:
        train_vd_list = json.load(f)
    vd_name_list = np.array(train_vd_list['train'])

    # read data (missing mask)
    all_data = np.load(VD_TIME_MASK)

    # draw
    for i in range(all_data.shape[0]):
        draw_discrete_heatmap(vd_name_list, all_data, i)

if __name__ == '__main__':
    if not os.path.exists(FOLDER_PATH):
        os.mkdir(FOLDER_PATH)
    main()
