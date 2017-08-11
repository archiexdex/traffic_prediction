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

# FLAGs
IS_LOSSES_SAVED = True

# PATH
DATA_PATH = '/home/xdex/Desktop/traffic_flow_detection/taipei/training_data/'
FOLDER_PATH = '/home/jay/Desktop/traffic_flow_detection/taipei/find_outlier/AREA_2/'

# for plotting vd
VD_NAMES = []
# VD_NAMES = ['VMTG520', 'VMQGX40', 'VM7FI60', 'VLMG600',
#             'VLYGU40', 'VN5HV60', 'VN5HV61', 'VLRHT00']
GROUP_ID = '1'


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


def draw_data(vd_name, group_id, density, flow, speed, weights):
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
    z_result = generate_data_on_plane(weights, plane_input)

    # visualize fitted plane
    trace_plane = go.Surface(
        x=x_range,
        y=y_range,
        z=z_result,
        name='Regressor',
        opacity=0.3,
        showscale=False
    )

    # visualize raw data in 3d
    data = []
    data_per_trace = density.shape[0] // 10
    for i in range(10):
        idx = i * data_per_trace
        trace_raw = go.Scatter3d(
            x=density[idx: idx + data_per_trace],
            y=speed[idx: idx + data_per_trace],
            z=flow[idx: idx + data_per_trace],
            mode='markers',
            name='Raw Data Part: %d' % i,
            marker=dict(
                size=2,
                # colorscale='Viridis',
                # color=colors[idx: idx + data_per_trace],
                opacity=0.6
            )
        )
        data.append(trace_raw)
    data.append(trace_plane)

    layout = go.Layout(
        title='VD: %s, GROUP_ID: %s.html' % (vd_name, group_id),
        autosize=True
    )

    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename=FOLDER_PATH +
                        'VD: %s, GROUP_ID: %s.html' % (vd_name, group_id))


def lsq_regressor(raw_data, vd_name, group_id, if_vis=False):
    """
    1. find regressor
    2. visaulize in 3D (if_vis = True)
    Params:
        raw_data: json, get data list by raw_data[vd_name][group_id]
        vd_name: string, e.g. 'VMTG520'
        group_id: string, e.g. '1'
        if_vis: bool, if True-> draw in 3D; if False-> only find regressor
    Saved Files:
        filename format: 'VD: vd_name, GROUP_ID: group_id.html'
    Return:
        weight: float, shape=[6,]
        losses: float, shape=[nums_data,]
    """
    target_vd_data = np.array(raw_data[vd_name][group_id])[:, 0:5]  # dfswt
    density = target_vd_data[:, 0]
    flow = target_vd_data[:, 1]
    speed = target_vd_data[:, 2]
    train_data = np.stack([density, speed])
    train_label = flow

    # regressor
    def func(W, inputs, labels):
        x = inputs[0]  # density
        y = inputs[1]  # speed
        z = labels  # flow
        return W[0] + W[1] * x + W[2] * y + W[3] * x * x + W[4] * x * y + W[5] * y * y - z

    W = np.ones(6)
    res_lsq = least_squares(func, W, args=(train_data, train_label))
    weights = res_lsq.x
    # print("weight:", weights)

    # compute losses, distance of data to plane
    losses = func(weights, train_data, train_label)

    # visulization in 3D
    if if_vis:
        draw_data(vd_name, group_id, density, flow, speed, weights)

    return weights, losses


def main():
    if not IS_LOSSES_SAVED:
        # load raw data
        raw_filename = DATA_PATH + 'fix_raw_data.json'
        with codecs.open(raw_filename, 'r', 'utf-8') as f:
            raw_data = json.load(f)

        # # plot
        for vd_name in VD_NAMES:
            for group_id in raw_data[vd_name]:
                lsq_regressor(raw_data, vd_name, group_id, if_vis=True)
                print("Finished:: VD_NAME: %s, GROUP_ID: %s" %
                      (vd_name, group_id))

        # # remove outliers
        # 1. find plane by regressor
        # 2. get loss from data to plane
        # 3. remove losses where data > mean + 2*stddev or data < mean -
        # 2*stddev
        all_losses = []
        for vd_name in raw_data:
            for group_id in raw_data[vd_name]:
                weights, losses = lsq_regressor(
                    raw_data, vd_name, group_id, if_vis=False)
                print("Finished:: VD_NAME: %s, GROUP_ID: %s" %
                      (vd_name, group_id))
                all_losses.append(losses)
        np.save('all_losses.npy', all_losses)
    else:
        all_losses = np.load('all_losses.npy')
        print('all_losses.shape', all_losses.shape)
        mean = np.mean(all_losses)
        stddev = np.std(all_losses)
        print('mean:', mean)
        print('stddev:', stddev)
        count = np.sum(all_losses > (mean + 1 * stddev)) + \
            np.sum(all_losses < (mean - 1 * stddev))
        print('count:', count)

        # visulization
        res_x = np.reshape(
            all_losses, all_losses.shape[0] * all_losses.shape[1])
        nums_data = res_x.shape[0]
        # grouping data for performance concern
        divider = 1000
        vis_x = []
        for i in range(nums_data // divider):
            idx = i * divider
            vis_x.append(np.mean(res_x[idx:idx + divider]))
        vis_x = np.array(vis_x)
        print(vis_x.shape)
        data = [
            go.Histogram(
                x=vis_x,
                histnorm='probability'
            )
        ]
        plotly.offline.plot(data, filename='histogram_of_losses.html')


if __name__ == '__main__':
    if not os.path.exists(FOLDER_PATH):
        os.mkdir(FOLDER_PATH)
    main()
