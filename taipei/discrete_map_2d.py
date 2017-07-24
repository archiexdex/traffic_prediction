from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import codecs
import json
import matplotlib.pyplot as plt


def visulize_map(row, col, vd_discrete_list):
    image = np.zeros(shape=[row, col], dtype=np.int8)
    for _, v in enumerate(vd_discrete_list):
        image[v[0], v[1]] = 255
    plt.figure(0)
    plt.figimage(image, resize=True)
    plt.savefig('discrete_map.png')
    # plt.show()


def discrete_map(row, col, vd_gps_dict):
    """
    Params:
        row, col:
        vd_gps_dict: json, {'key':'value'->'vd_id':'[gps]'}
    Return:
        vd_discrete_dict: json,
    """
    vd_gps_list = np.array(list(vd_gps_dict.values()))
    lon_max = np.amax(vd_gps_list[:, 0])
    lon_min = np.amin(vd_gps_list[:, 0])
    lat_max = np.amax(vd_gps_list[:, 1])
    lat_min = np.amin(vd_gps_list[:, 1])
    lon_range = abs(lon_max - lon_min + 0.000001)
    lat_range = abs(lat_max - lat_min + 0.000001)

    vd_discrete_dict = {}
    for _, v in enumerate(vd_gps_dict):
        lon_discrete = int(abs(vd_gps_dict[v][0] - lon_min) / lon_range * col)
        lat_discrete = int(abs(vd_gps_dict[v][1] - lat_min) / lat_range * row)
        # print("(longitude, latitude): (%d, %d)" % (lon_discrete, lat_discrete))
        if [lon_discrete, lat_discrete] in vd_discrete_dict.values():
            print("!!!!!!!!!!!!duplicated!!!!!!!!!!!!")
        vd_discrete_dict[v] = [lon_discrete, lat_discrete]

    return vd_discrete_dict


def check_duplicate(vd_discrete_list):
    """
    check if the discreted map have duplicated location or not
    """
    sorted_list = sorted(vd_discrete_list)
    last_v = [-1, -1]
    for _, v in enumerate(vd_discrete_list):
        if last_v == v:
            return False
        last_v = v

    return True


def main():
    """
    main
    """
    # 1. load raw_data {'key': 'value' -> 'vd_id': '[intervals, features]'}
    raw_data = np.load(
        '/home/xdex/Desktop/traffic_flow_detection/taipei/fix_input_data.npy')
    raw_data = raw_data.item()
    # get size
    target_map_rows = 32
    target_map_cols = 32
    num_intervals = np.array(raw_data['VMTG520']).shape[0]
    print('num_intervals', num_intervals)
    num_features = np.array(raw_data['VMTG520']).shape[1]
    # 2. load gps_data {'key': 'value' -> 'vd_id': '[gps]'}
    raw_gps_data = np.load(
        '/home/xdex/Desktop/traffic_flow_detection/taipei/VD_GPS.npy')
    raw_gps_data = raw_gps_data.item()
    # 3. generate smallest discrete map from gps_data
    # read vd_list.txt
    vd_list = []
    with open('/home/xdex/Desktop/traffic_flow_detection/taipei/vd_list') as f:
        for i in f:
            key = i.strip()
            vd_list.append(key)
    vd_gps_dict = {}
    for v in vd_list:
        vd_gps_dict[v] = raw_gps_data[v]
    discreted_gps_dict = discrete_map(
        target_map_rows, target_map_cols, vd_gps_dict)
    with codecs.open('discreted_gps_dict.json', 'w', 'utf-8') as out:
        json.dump(discreted_gps_dict, out, ensure_ascii=False)
    # 4. discreted_map = zeros[row, col, intervals, features]
    discreted_map = []
    # visulize_map(target_map_rows, target_map_cols,
    #              list(discreted_gps_dict.values()))
    # 5. discreted_map[discrete_longitude][discrete_latitude] = raw_data['vd_id']
    for _, v in enumerate(discreted_gps_dict):
        print(np.array(raw_data[v]).shape)
        if 230976 == np.array(raw_data[v]).shape[0]:  # TODO!!!!!!!!!!
            discreted_map.append(raw_data[v])
    discreted_map = np.array(discreted_map)
    discreted_map = np.transpose(discreted_map, [1, 0, 2])
    print(discreted_map.shape)
    # 6. generate data from shape=[row, col, intervals, features] to shape=[num_data, row, col, 12, features]
    repeated_data = []
    # iter all interval
    for i in range(discreted_map.shape[0] - 12):
        repeated_data.append(discreted_map[i:i + 12, :, :])
    repeated_data = np.array(repeated_data)
    print(repeated_data.shape)
    # 7. generate label data from shape=[row, col, intervals, features] to shape=[num_data, num_vd]
    label_data = []
    for i in range(12, discreted_map.shape[0]):  # iter all interval
        label_data.append(discreted_map[i, :, 1])  # 1-> flow only
    label_data = np.array(label_data)
    print(label_data.shape)
    # 8. split data into 9:1 as num_train_data:num_test_data
    train_data, test_data = np.split(
        repeated_data, [repeated_data.shape[0] * 9 // 10])
    np.save('train_data.npy', train_data)
    np.save('test_data.npy', test_data)
    print('data saved')
    train_label, test_label = np.split(
        label_data, [label_data.shape[0] * 9 // 10])
    np.save('train_label.npy', train_label)
    np.save('test_label.npy', test_label)
    print('label saved')


if __name__ == '__main__':
    main()
