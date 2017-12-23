"""
To show selected area with train and label
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
VD_LIST_PATH = '/home/xdex/Desktop/traffic_flow_detection/taipei/training_data/old_Taipei_data/vd_base/target_vd_list.json'
VD_GPS_FILE = '/home/xdex/Desktop/traffic_flow_detection/taipei/training_data/new_raw_data/VD_GPS.npy'
MISSING_MASK_PATH = os.path.join(DATA_PATH, 'mask_data')
OUTLIER_MASK_PATH = os.path.join(
    DATA_PATH, 'mask_outlier')


def draw_heatmap(vd_list, vd_gps_dict, missing_dict, outliers_dict, both_dict):
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
    data_dict["train"] = {}
    data_dict["train"]['missing'] = []
    data_dict["train"]['outlier'] = []
    data_dict["train"]['both'] = []
    data_dict["train"]['lat'] = []
    data_dict["train"]['lon'] = []
    data_dict["train"]['vd_name'] = []

    data_dict["label"] = {}
    data_dict["label"]['missing'] = []
    data_dict["label"]['outlier'] = []
    data_dict["label"]['both'] = []
    data_dict["label"]['lat'] = []
    data_dict["label"]['lon'] = []
    data_dict["label"]['vd_name'] = []
    for vd_name in vd_list['total']:
        print(vd_name)
        for group_id in range(5):
            try:
                if vd_name in vd_list['label']:
                    data_dict["label"]['missing'].append(
                        missing_dict[vd_name + '_%s' % group_id])
                    data_dict["label"]['outlier'].append(
                        outliers_dict[vd_name + '_%s' % group_id])
                    data_dict["label"]['both'].append(both_dict[vd_name + '_%s' % group_id])
                    data_dict["label"]['lat'].append(vd_gps_dict[vd_name][0])
                    data_dict["label"]['lon'].append(
                        vd_gps_dict[vd_name][1] + group_id * 0.0001)
                    data_dict["label"]['vd_name'].append(vd_name + '_%s' % group_id)
                else:                    
                    data_dict["train"]['missing'].append(
                        missing_dict[vd_name + '_%s' % group_id])
                    data_dict["train"]['outlier'].append(
                        outliers_dict[vd_name + '_%s' % group_id])
                    data_dict["train"]['both'].append(both_dict[vd_name + '_%s' % group_id])
                    data_dict["train"]['lat'].append(vd_gps_dict[vd_name][0])
                    data_dict["train"]['lon'].append(
                        vd_gps_dict[vd_name][1] + group_id * 0.0001)
                    data_dict["train"]['vd_name'].append(vd_name + '_%s' % group_id)
            except:
                print('QQ cannot find vd: %s, grp: %s' % (vd_name, group_id))

    # prepare vis
    # color scale from blue to red
    scl = [[0.0, "rgb(255, 0, 0)"], [0.6, "rgb(255, 0, 0)"], [
        0.7, "rgb(255, 255, 0)"], [0.8, "rgb(0, 255, 0)"], [0.9, "rgb(0, 255, 255)"], [1.0, "rgb(0, 0, 255)"], ]
    # traces
    description_list = {
        "train":[],
        "label":[]
    }
    for vd_name_g, num_missing, num_outlier, num_both in zip( 
            data_dict["train"]['vd_name'], data_dict["train"]['missing'], data_dict["train"]['outlier'], data_dict["train"]['both']):
        description = 'train VD: %s, missing: %f %%, outlier: %f %%, both: %f %%' % (
            vd_name_g, num_missing, num_outlier, num_both)
        description_list["train"].append(description)

    for vd_name_g, num_missing, num_outlier, num_both in zip( 
            data_dict["label"]['vd_name'], data_dict["label"]['missing'], data_dict["label"]['outlier'], data_dict["label"]['both']):
        description = 'label VD: %s, missing: %f %%, outlier: %f %%, both: %f %%' % (
            vd_name_g, num_missing, num_outlier, num_both)
        description_list["label"].append(description)
    
    
    train_missing_trace = Scattermapbox(
        name='train_missing rate',
        lat=data_dict["train"]['lat'],
        lon=data_dict["train"]['lon'],
        mode='markers',
        marker=Marker(
            size=10,
            opacity=0.8,
            reversescale=True,
            autocolorscale=False,
            colorscale=scl,
            cmin=0,
            color=data_dict["train"]['missing'],
            cmax=100,
            colorbar=dict(
                len=0.8
            )
        ),
        text=description_list["train"]
    )
    train_outlier_trace = Scattermapbox(
        name='outlier rate',
        lat=data_dict["train"]['lat'],
        lon=data_dict["train"]['lon'],
        mode='markers',
        marker=Marker(
            size=10,
            opacity=0.8,
            reversescale=True,
            autocolorscale=False,
            colorscale=scl,
            cmin=0,
            color=data_dict["train"]['outlier'],
            cmax=100,
            colorbar=dict(
                len=0.8
            )
        ),
        text=description_list["train"]
    )
    train_both_trace = Scattermapbox(
        name='both rate',
        lat=data_dict["train"]['lat'],
        lon=data_dict["train"]['lon'],
        mode='markers',
        marker=Marker(
            size=10,
            opacity=0.8,
            reversescale=True,
            autocolorscale=False,
            colorscale=scl,
            cmin=0,
            color=data_dict["train"]['both'],
            cmax=100,
            colorbar=dict(
                len=0.8
            )
        ),
        text=description_list["train"]
    )

    label_missing_trace = Scattermapbox(
        name='label_missing rate',
        lat=data_dict["label"]['lat'],
        lon=data_dict["label"]['lon'],
        mode='markers',
        marker=Marker(
            size=20,
            opacity=0.8,
            reversescale=True,
            autocolorscale=False,
            colorscale=scl,
            cmin=0,
            color=data_dict["label"]['missing'],
            cmax=100,
            colorbar=dict(
                len=0.8
            )
        ),
        text=description_list["label"]
    )
    label_outlier_trace = Scattermapbox(
        name='outlier rate',
        lat=data_dict["label"]['lat'],
        lon=data_dict["label"]['lon'],
        mode='markers',
        marker=Marker(
            size=20,
            opacity=0.8,
            reversescale=True,
            autocolorscale=False,
            colorscale=scl,
            cmin=0,
            color=data_dict["label"]['outlier'],
            cmax=100,
            colorbar=dict(
                len=0.8
            )
        ),
        text=description_list["label"]
    )
    label_both_trace = Scattermapbox(
        name='both rate',
        lat=data_dict["label"]['lat'],
        lon=data_dict["label"]['lon'],
        mode='markers',
        marker=Marker(
            size=20,
            opacity=0.8,
            reversescale=True,
            autocolorscale=False,
            colorscale=scl,
            cmin=0,
            color=data_dict["label"]['both'],
            cmax=100,
            colorbar=dict(
                len=0.8
            )
        ),
        text=description_list["label"]
    )
    data = [train_missing_trace, train_outlier_trace, train_both_trace,
            label_missing_trace, label_outlier_trace, label_both_trace]
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
        fig, filename='statistics of missing data and outliers.html')
    print('file saved: statistics of missing data and outliers.html')
    return


# # Parameters
MAP_ROWS = 30
MAP_COLS = 30


def draw_train_label(vd_list, vd_gps_dict):
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
    data_dict = {}
    data_dict['train'] = []
    data_dict['label'] = []
    
    data_dict['lat'] = []
    data_dict['lon'] = []
    data_dict['vd_name'] = []

    

    for vd_name in vd_list['train']:
        print(vd_name)
        for group_id in range(5):
            try:
                data_dict['train'].append(1)
                if vd_name in vd_list['label']:
                    data_dict['label'].append(0)
                
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
    
    train_trace = Scattermapbox(
        name='train',
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
            color=data_dict['train'],
            cmax=100,
            colorbar=dict(
                len=0.8
            )
        )
        
    )
    label_trace = Scattermapbox(
        name='label',
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
            color=data_dict['label'],
            cmax=100,
            colorbar=dict(
                len=0.8
            )
        )
        
    )
    
    data = [train_trace, outlier_trace]
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
        fig, filename='selected_area.html')
    print('selected_area.html')
    return


def main():
    
    if IS_DRAW_DISCRETE:
        # step 1
        vd_gps_dict = np.load(VD_GPS_FILE).item()
        with codecs.open(VD_LIST_PATH, 'r', 'utf-8') as fp:
            vd_list = json.load(fp)

        draw_train_label(vd_list, vd_gps_dict)
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
                missing_statistics[key_name] = np.sum(
                    missing_mask) / missing_mask.shape[0] * 100
                outlier_statistics[key_name] = np.sum(
                    outlier_mask) / outlier_mask.shape[0] * 100
                both_statistics[key_name] = np.sum(
                    np.logical_or(missing_mask, outlier_mask)) / missing_mask.shape[0] * 100

        # step 4
        vd_list = {}
        with open(VD_LIST_PATH) as fp:
            vd_list = json.load(fp)
        draw_heatmap(vd_list, vd_gps_dict, missing_statistics,
                     outlier_statistics, both_statistics)


if __name__ == '__main__':
    main()
