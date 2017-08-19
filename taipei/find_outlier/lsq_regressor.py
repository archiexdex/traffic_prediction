"""
To find outliers in every vds.
output the outlier mask per vd.
visualize them if needed.
"""


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
OUTLIERS_THRSHOLD = 1

# PATH
DATA_PATH = '/home/xdex/Desktop/traffic_flow_detection/taipei/training_data/new_raw_data/vd_base/5/fix_data_grp/'
MASK_SAVED_PATH = '/home/xdex/Desktop/traffic_flow_detection/taipei/training_data/new_raw_data/vd_base/5/outlier_mask_grp/'
FOLDER_PATH = '/home/jay/Desktop/traffic_flow_detection/taipei/find_outlier/AREA_2/'


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


def draw_data(vd_name, density, flow, speed, weights):
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
        title='VD: %s.html' % (vd_name),
        autosize=True
    )

    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename=FOLDER_PATH +
                        'VD: %s.html' % (vd_name))


def lsq_regressor(raw_data, vd_name, if_vis=False, outlier_mask=None):
    """
    1. find regressor
    2. visaulize in 3D (if_vis = True)
    Params:
        raw_data: TODO
        vd_name: string, e.g. 'VMTG520_0'
        if_vis: bool, if True-> draw in 3D; if False-> only find regressor
        outlier_mask(optional): bool,
    Saved Files:
        filename format: 'VD: vd_name.html'
    Return:
        weight: float, shape=[6,]
        losses: float, shape=[nums_data,]
    """
    density = raw_data[:, 1]
    # normalization
    if np.std(density) == 0.0:
        density = np.zeros_like(density)
    else:
        density = np.divide(density - np.mean(density), np.std(density))
    flow = raw_data[:, 2]
    speed = raw_data[:, 3]
    # normalization
    if np.std(speed) == 0.0:
        speed = np.zeros_like(speed)
    else:
        speed = np.divide(speed - np.mean(speed), np.std(speed))

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
        draw_data(vd_name, density, flow, speed, weights)

    return weights, losses.tolist()


def main():
    # n = 10
    all_losses_dict = {}
    for path, _, file_names in os.walk(DATA_PATH):
        for file_name in file_names:
            file_path = os.path.join(path, file_name)
            vd_data = np.load(file_path)
            _, losses = lsq_regressor(
                vd_data, file_name[:-4], if_vis=False)  # [:-4]: remove '.npy'
            all_losses_dict[file_name] = losses
            # n -= 1
            # if n == 0:
            #     break
    all_losses_np = np.array(list(all_losses_dict.values()))
    print('all_losses_np.shape', all_losses_np.shape)
    print('all_losses_np.dtype', all_losses_np.dtype)
    mean = np.mean(all_losses_np)
    stddev = np.std(all_losses_np)
    num_outliers = np.sum(all_losses_np > (mean + OUTLIERS_THRSHOLD * stddev)) + \
        np.sum(all_losses_np < (mean - OUTLIERS_THRSHOLD * stddev))

    for vd_name in all_losses_dict:
        outlier_mask = np.logical_or(all_losses_dict[vd_name] > (
            mean + OUTLIERS_THRSHOLD * stddev),  all_losses_dict[vd_name] < (mean - OUTLIERS_THRSHOLD * stddev))
        np.save(os.path.join(MASK_SAVED_PATH, vd_name), outlier_mask)
        print('%s Saved' % (os.path.join(MASK_SAVED_PATH, vd_name)))

    # log
    print('mean:', mean)
    print('stddev:', stddev)
    print('num_outliers:', num_outliers)
    print('removed_rate: %f %%' % (100 * num_outliers /
                                   (all_losses_np.shape[0] * all_losses_np.shape[1])))


if __name__ == '__main__':
    if not os.path.exists(FOLDER_PATH):
        os.mkdir(FOLDER_PATH)
    if not os.path.exists(MASK_SAVED_PATH):
        os.mkdir(MASK_SAVED_PATH)
    main()
