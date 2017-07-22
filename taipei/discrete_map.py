from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt


def visulize_map(row, col, vd_discrete_list):
    image = np.zeros(shape=[row, col], dtype=np.int8)
    for _, v in enumerate(vd_discrete_list):
        image[v[0], v[1]] = 255
    plt.figure(0)
    plt.figimage(image, resize=True)
    plt.show()

def discrete_map(row, col, vd_gps_list):
    """
    Params:
        row, col:
        vd_gps_list:
    Return:
    """
    lon_max = np.amax(vd_gps_list[:, 0])
    lon_min = np.amin(vd_gps_list[:, 0])
    lat_max = np.amax(vd_gps_list[:, 1])
    lat_min = np.amin(vd_gps_list[:, 1])
    lon_range = abs(lon_max - lon_min + 0.000001)
    lat_range = abs(lat_max - lat_min + 0.000001)

    vd_discrete_list = []
    for i, v in enumerate(vd_gps_list):
        lon_discrete = int(abs(v[0] - lon_min) / lon_range * col)
        lat_discrete = int(abs(v[1] - lat_min) / lat_range * row)
        print("(longitude, latitude): (%d, %d)" % (lon_discrete, lat_discrete))
        if [lon_discrete, lat_discrete] in vd_discrete_list:
            print("!!!!!!!!!!!!duplicated!!!!!!!!!!!!")
        vd_discrete_list.append([lon_discrete, lat_discrete])

    return vd_discrete_list


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
    raw_data = np.load('file_name.npy')
    raw_data=raw_data.item()
    # get size
    target_map_rows = 64
    target_map_cols = 64
    num_intervals = raw_data['VMTG520'].shape[0]
    num_features = raw_data['VMTG520'].shape[1]
    # 2. load gps_data {'key': 'value' -> 'vd_id': '[gps]'}
    raw_gps_data = np.load('file_name.npy')
    raw_gps_data = raw_gps_data.item()
    # 3. generate smallest discrete map from gps_data
    # read vd_list.txt
    vd_list = []
    with open('vd_list.txt') as f:
        vd_list.append(f.readline())
    vd_gps_list = []
    for v in vd_list:
        vd_gps_list.append(raw_gps_data[v])
    discreted_gps_list = discrete_map(target_map_rows, target_map_cols, vd_gps_list)
    # 4. discreted_map = zeros[row, col, intervals, features]
    discreted_map = np.zeros(shape=[target_map_rows, target_map_cols, num_intervals, num_features])
    # 5. discreted_map[discrete_longitude][discrete_latitude] = raw_data['vd_id']
    for v in discreted_gps_list:
        discreted_map[v[0]][v[1]] = raw_data[]
    # 6. generate data from shape=[row, col, intervals, features] to shape=[num_data, row, col, 12, features]
    # 7. split data into 9:1 as num_train_data:num_test_data

def test():
    """
    test
    """
    vd_gps_list = []
    vd_gps_list.append([123.123123, 23.345123])
    vd_gps_list.append([123.321321, 23.345123])
    vd_gps_list.append([123.123123, 23.543543])
    vd_gps_list.append([123.345345, 23.876876])
    vd_gps_list = np.array(vd_gps_list)
    print(vd_gps_list.shape)

    discreted_list = discrete_map(10, 10, vd_gps_list)
    print(check_duplicate(discreted_list))

    visulize_map(10, 10, discreted_list)


if __name__ == '__main__':
    # test()
    main()