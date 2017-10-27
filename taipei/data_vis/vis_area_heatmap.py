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

# MAPBOX
ACCESS_TOKEN = 'pk.eyJ1IjoiY2h5Y2hlbiIsImEiOiJjajZoaWo5aHYwNm44MnF0ZW56MTljaGp1In0.T6HvF9dyb9YS2ptOzjcD5A'

# FLAGS
IS_DRAW_DISCRETE = False
IS_NEW_DATA = False

# PATH
if IS_NEW_DATA:
    DATA_PATH = '/home/xdex/Desktop/traffic_flow_detection/taipei/training_data/new_raw_data/vd_base/5/'
else:
    DATA_PATH = '/home/xdex/Desktop/traffic_flow_detection/taipei/training_data/old_Taipei_data/vd_base/'
VD_GPS_FILE = '/home/xdex/Desktop/traffic_flow_detection/taipei/training_data/new_raw_data/VD_GPS.npy'
MISSING_MASK_PATH = os.path.join(DATA_PATH, 'mask_data')
OUTLIER_MASK_PATH = os.path.join(
    DATA_PATH, 'mask_outlier')


def draw_heatmap(vd_gps_dict, missing_dict, outliers_dict, both_dict):
    """draw heatmap by the rate of each type of statistics, including mssing data rate, outliers rate, both above attribute rate

    Params
    ------
    vd_gps_dict : json
        gps map, key == vd_name, value == [lat, lon]
    missing_dict : json
        missing data statistics, key == vd_name, value == rate of missing data
    outliers_dict : json
        outliers statistics, key == vd_name, value == rate of outlier
    both_dict : json
        statistics of both outlier or missing, key == vd_name, value == rate of problem data

    Return
    ------
    nope

    Raises
    ------
    nope
    """
    data_dict = {}
    data_dict['missing'] = []
    data_dict['outlier'] = []
    data_dict['both'] = []
    data_dict['lat'] = []
    data_dict['lon'] = []
    data_dict['vd_name'] = []
    for vd_name in vd_gps_dict:
        for group_id in range(5):
            try:
                data_dict['missing'].append(
                    missing_dict[vd_name + '_%s' % group_id])
                data_dict['outlier'].append(
                    outliers_dict[vd_name + '_%s' % group_id])
                data_dict['both'].append(both_dict[vd_name + '_%s' % group_id])
                data_dict['lat'].append(vd_gps_dict[vd_name][0])
                data_dict['lon'].append(
                    vd_gps_dict[vd_name][1] + group_id * 0.0001)
                data_dict['vd_name'].append(vd_name + '_%s' % group_id)
            except:
                print('QQ cannot find vd: %s, grp: %s' % (vd_name, group_id))

    # prepare vis
    # color scale from blue to red
    scl = [[0.0, "rgb(255, 0, 0)"], [0.6, "rgb(255, 0, 0)"], [
        0.7, "rgb(255, 255, 0)"], [0.8, "rgb(0, 255, 0)"], [0.9, "rgb(0, 255, 255)"], [1.0, "rgb(0, 0, 255)"], ]
    # traces
    description_list = []
    for vd_name_g, num_missing, num_outlier, num_both in zip(data_dict['vd_name'], data_dict['missing'], data_dict['outlier'], data_dict['both']):
        description = 'VD: %s, missing: %f %%, outlier: %f %%, both: %f %%' % (
            vd_name_g, num_missing, num_outlier, num_both)
        description_list.append(description)
    missing_trace = Scattermapbox(
        name='missing rate',
        lat=data_dict['lat'],
        lon=data_dict['lon'],
        mode='markers',
        marker=Marker(
            size=10,
            opacity=0.8,
            reversescale=True,
            autocolorscale=False,
            colorscale=scl,
            cmin=0,
            color=data_dict['missing'],
            cmax=100,
            colorbar=dict(
                len=0.8
            )
        ),
        text=description_list
    )
    outlier_trace = Scattermapbox(
        name='outlier rate',
        lat=data_dict['lat'],
        lon=data_dict['lon'],
        mode='markers',
        marker=Marker(
            size=10,
            opacity=0.8,
            reversescale=True,
            autocolorscale=False,
            colorscale=scl,
            cmin=0,
            color=data_dict['outlier'],
            cmax=100,
            colorbar=dict(
                len=0.8
            )
        ),
        text=description_list
    )
    both_trace = Scattermapbox(
        name='both rate',
        lat=data_dict['lat'],
        lon=data_dict['lon'],
        mode='markers',
        marker=Marker(
            size=10,
            opacity=0.8,
            reversescale=True,
            autocolorscale=False,
            colorscale=scl,
            cmin=0,
            color=data_dict['both'],
            cmax=100,
            colorbar=dict(
                len=0.8
            )
        ),
        text=description_list
    )
    data = [missing_trace, outlier_trace, both_trace]
    layout = Layout(
        autosize=True,
        hovermode='closest',
        mapbox=dict(
            accesstoken=ACCESS_TOKEN,
            bearing=0,
            # Taipei Train Station
            center=dict(
                lat=25.046353,
                lon=121.517586
            ),
            pitch=0,
            zoom=10
        ),
        xaxis=dict(
            rangeslider=dict(),
            type='date'
        )
    )

    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(
        fig, filename='all statistics of missing data and outliers.html')
    print('file saved: statistics of missing data and outliers.html')
    return


# # Parameters
MAP_ROWS = 30
MAP_COLS = 30


def draw_discrete_heatmap(vd_loc_dict, missing_dict, outliers_dict, both_dict):
    """draw heatmap by the rate of each type of statistics, including mssing data rate, outliers rate, both above attribute rate

    Params
    ------
    vd_loc_dict : json
        som map, key == vd_name, value == [x, y]
    missing_dict : json
        missing data statistics, key == vd_name, value == rate of missing data
    outliers_dict : json
        outliers statistics, key == vd_name, value == rate of outlier
    both_dict : json
statistics of both outlier or missing, key == vd_name, value == rate
of problem data

    Return
    ------
    nope

    Raises
    ------
    nope
    """
    missing_map = np.zeros(shape=[MAP_ROWS, 2 * MAP_COLS])
    outlier_map = np.zeros(shape=[MAP_ROWS, 2 * MAP_COLS])
    both_map = np.zeros(shape=[MAP_ROWS, 2 * MAP_COLS])
    for key in vd_loc_dict:
        loc = vd_loc_dict[key]
        print(key)
        try:
            missing_map[loc[0], 2 * loc[1]] = missing_dict[key + '_0']
            outlier_map[loc[0], 2 * loc[1]] = outliers_dict[key + '_0']
            both_map[loc[0], 2 * loc[1]] = both_dict[key + '_0']
        except:
            print('QQ cannot find vd %s group0' % key)
        try:
            missing_map[loc[0], 2 * loc[1] + 1] = missing_dict[key + '_1']
            outlier_map[loc[0], 2 * loc[1] + 1] = outliers_dict[key + '_1']
            both_map[loc[0], 2 * loc[1] + 1] = both_dict[key + '_1']
        except:
            print('QQ cannot find vd %s group1' % key)

    # prepare vis
    missing_trace = go.Heatmap(
        zauto=False, z=missing_map, name='missing rate', colorbar=dict(len=0.33, y=0.85))
    outlier_trace = go.Heatmap(
        zauto=False, z=outlier_map, name='outlier rate', colorbar=dict(len=0.33, y=0.5))
    both_trace = go.Heatmap(zauto=False, z=both_map,
                            name='both rate', colorbar=dict(len=0.33, y=0.15))
    data = [missing_trace, outlier_trace, both_trace]

    fig = plotly.tools.make_subplots(
        rows=3, cols=1, shared_xaxes=True, shared_yaxes=True)
    fig.append_trace(missing_trace, 1, 1)
    fig.append_trace(outlier_trace, 2, 1)
    fig.append_trace(both_trace, 3, 1)
    fig['layout'].update(title='statistics of missing data and outliers',
                         autosize=True)

    plotly.offline.plot(
        fig, filename='statistics of missing data and outliers.html')


def main():
    if IS_DRAW_DISCRETE:
        # step 1
        som_filename = os.path.join(DATA_PATH, 'som_list.json')
        with codecs.open(som_filename, 'r', 'utf-8') as f:
            vd_loc_dict = json.load(f)

        # step 2, 3
        missing_statistics = {}
        outlier_statistics = {}
        both_statistics = {}
        for _, _, file_names in os.walk(MISSING_MASK_PATH):
            for file_name in file_names:
                missing_mask_path = os.path.join(MISSING_MASK_PATH, file_name)
                outlier_mask_path = os.path.join(OUTLIER_MASK_PATH, file_name)
                missing_mask = np.load(missing_mask_path)
                outlier_mask = np.load(outlier_mask_path)
                key_name = file_name[:-4]  # remove '.npy'
                missing_statistics[key_name] = np.sum(
                    missing_mask) / missing_mask.shape[0] * 100
                outlier_statistics[key_name] = np.sum(
                    outlier_mask) / outlier_mask.shape[0] * 100
                both_statistics[key_name] = np.sum(
                    np.logical_or(missing_mask, outlier_mask)) / missing_mask.shape[0] * 100

        # step 4
        draw_discrete_heatmap(vd_loc_dict, missing_statistics,
                              outlier_statistics, both_statistics)
    else:
        # step 1. load gps map
        # step 2. read masks
        # step 3. statistic missing data, outliers rate per vd
        # step 4. visulize above three tag with heat map

        # step 1
        vd_gps_dict = np.load(VD_GPS_FILE).item()

        # step 2, 3
        missing_statistics = {}
        outlier_statistics = {}
        both_statistics = {}
        for _, _, file_names in os.walk(MISSING_MASK_PATH):
            for file_name in file_names:
                missing_mask_path = os.path.join(MISSING_MASK_PATH, file_name)
                outlier_mask_path = os.path.join(OUTLIER_MASK_PATH, file_name)
                missing_mask = np.load(missing_mask_path)
                outlier_mask = np.load(outlier_mask_path)
                key_name = file_name[:-4]  # remove '.npy'
                # if np.sum( np.logical_or(missing_mask, outlier_mask)) / missing_mask.shape[0] * 100 > 10:
                #     continue
                missing_statistics[key_name] = np.sum(
                    missing_mask) / missing_mask.shape[0] * 100
                outlier_statistics[key_name] = np.sum(
                    outlier_mask) / outlier_mask.shape[0] * 100
                both_statistics[key_name] = np.sum(
                    np.logical_or(missing_mask, outlier_mask)) / missing_mask.shape[0] * 100

        # step 4
        draw_heatmap(vd_gps_dict, missing_statistics,
                     outlier_statistics, both_statistics)


if __name__ == '__main__':
    main()
