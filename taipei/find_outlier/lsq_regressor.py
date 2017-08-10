from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import codecs
import json
import datetime
import numpy as np
import plotly
plotly.__version__
plotly.tools.set_credentials_file(
    username='ChenChiehYu', api_key='xh9rsxFXY6DNF1qAfUyQ')
import plotly.plotly as py
import plotly.graph_objs as go
import pandas as pd
# regression
from scipy.optimize import least_squares

DATA_PATH = '/home/xdex/Desktop/traffic_flow_detection/taipei/training_data/'
FOLDER_PATH = '/home/jay/Desktop/traffic_flow_detection/taipei/find_outlier/test/'
VD_NAME = 'VMQGX40' # bad vd
# VD_NAME = 'VNNFY00' # good vd
#TODO automatically generate vis result accordint to list of vds
GROUP_ID = '1'


def func(weights, inputs, labels):
    x = inputs[0]  # density
    y = inputs[1]  # speed
    z = labels  # flow
    return weights[0] + weights[1] * x + weights[2] * y + weights[3] * x * x + weights[4] * x * y + weights[5] * y * y - z


def generate_data_on_plane(weights, inputs):
    """
    Params:
        inputs: float, shape=[row, col, 2], density and speed
    Return:
        outputs: float, shape=[row, col], flow
    """
    output = []
    for _, rows in enumerate(inputs):
        temp = []
        for _, features in enumerate(rows):
            x = features[0]  # density
            y = features[1]  # speed
            z = weights[0] + weights[1] * x + weights[2] * y + \
                weights[3] * x * x + weights[4] * x * y + weights[5] * y * y
            temp.append(z)
        output.append(temp)
    output = np.array(output)
    return output


def main():
    # load raw data
    raw_filename = DATA_PATH + 'fix_raw_data_time1.json'
    with codecs.open(raw_filename, 'r', 'utf-8') as f:
        raw_data = json.load(f)

    target_vd_data = np.array(raw_data[VD_NAME][GROUP_ID])[:, 0:3]  # dfs
    density = target_vd_data[:, 0]
    flow = target_vd_data[:, 1]
    speed = target_vd_data[:, 2]
    train_data = np.stack([density, speed])
    train_label = flow

    # regressor
    W = np.ones(6)
    res_lsq = least_squares(func, W, args=(train_data, train_label))
    print("weight:", res_lsq.x)

    # find density range and speed range
    density_max = np.amax(density)
    density_min = np.amin(density)
    x_range = np.linspace(density_min, density_max, num=100)
    x_range = [x_range for _ in range(100)]

    speed_max = np.amax(speed)
    speed_min = np.amin(speed)
    y_range = np.linspace(speed_min, speed_max, num=100)
    y_range = [y_range for _ in range(100)]
    y_range = np.transpose(y_range)

    plane_input = np.stack([x_range, y_range])
    plane_input = np.transpose(plane_input, [1, 2, 0])

    # get flow value on fitted plane
    z_result = generate_data_on_plane(res_lsq.x, plane_input)

    # visualize fitted plane
    trace_plane = go.Surface(
        x=x_range,
        y=y_range,
        z=z_result,
        name='Regressor',
        opacity=0.3
    )

    # visualize raw data in 3d
    trace_raw = go.Scatter3d(
        x=density,
        y=speed,
        z=flow,
        mode='markers',
        name='Raw Data',
        marker=dict(
            size=2,
            opacity=0.6
        )
    )
    data = [
        trace_raw, trace_plane
    ]
    layout = go.Layout(
        title='VD: %s data scatter' % VD_NAME,
        autosize=True
    )

    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename=FOLDER_PATH +
                        'VD: %s data scatter.html' % VD_NAME)


if __name__ == '__main__':
    if not os.path.exists(FOLDER_PATH):
        os.mkdir(FOLDER_PATH)
    main()
