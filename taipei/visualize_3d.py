from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np


def main():
    # read data
    train_data = np.load('train_data.npy')
    # read vd_gps_dict
    with codecs.open('discreted_gps_dict.json', 'r', 'utf-8') as f:
        vd_gps_dict = json.load(f)

    # try visulize the first data
    train_data = train_data[0]

    verts = []
    for i, _ in enumerate(train_data):
        for ii, v in enumerate(vd_gps_dict):
            temp_row = vd_gps_dict[v][0]
            temp_col = vd_gps_dict[v][1]
            verts.append(zip(ii, train_data[temp_row][temp_col][i]))

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    def cc(arg):
        return mcolors.to_rgba((arg), alpha=0.6)

    # xs = np.arange(0, 10, 0.4)
    # verts = []
    # for z in range(12):
    #     ys = np.random.rand(len(xs))
    #     ys[0], ys[-1] = 0, 0
    #     verts.append(list(zip(xs, ys)))

    poly = PolyCollection(verts, facecolors=[cc('r'), cc('g'), cc('b'),
                                            cc('y')])
    poly.set_alpha(0.7)
    ax.add_collection3d(poly, zs=zs, zdir='y')

    ax.set_xlabel('X')
    ax.set_xlim3d(0, 10)
    ax.set_ylabel('Y')
    ax.set_ylim3d(-1, 4)
    ax.set_zlabel('Z')
    ax.set_zlim3d(0, 1)

    plt.show()

if __name__ == '__main__':
    main()
