from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random
import math
import matplotlib.pyplot as plot

sample_amount = 150
data_file = '/home/nctucgv/Documents/TrafficVis_Run/src/traffic_flow_detection/batch_no_over_data_mile_15_28.5_total_60_predict_1_20.npy'
sampled_file = 'is_'+str(sample_amount)+'_batch_no_over_data_mile_15_28.5_total_60_predict_1_20.npy'
sampled_dist_plot = 'is_'+str(sample_amount)+'.png'
is_PDF = False

def is_sampling():
    """
    importance weight = delta speed * mean flow
    maybe we could draw out the distribution to verify this hypothsis
    """
    input_data = np.load(data_file)
    print(input_data.shape)

    # sum over the whole input_data's weight by [speed*density]
    weight_sum = 0.0
    cdf = []
    pdf = []
    for _, v in enumerate(input_data):
        delta_speed = np.amax(v[:, :, 2]) - np.amin(v[:, :, 2])
        mean_flow = np.mean(v[:, :, 0])
        importance_weight = delta_speed**2 * mean_flow
        # print("delta_speed", delta_speed)
        # print("mean_flow", mean_flow)
        # print("importance_weight", importance_weight)
        weight_sum += importance_weight
        cdf.append(weight_sum)
        pdf.append(importance_weight)
    print("weight_sum", weight_sum)
    cdf = np.array(cdf)
    pdf = np.array(pdf)

    plot.figure(1)
    plot.hist(pdf, bins=10, normed=is_PDF)
    plot.savefig('ori.png')

    # random select non-repetitive index
    #random_index_list = np.arange(
    #    0, int(math.ceil(weight_sum)), step=1, dtype=np.int)
    #np.random.shuffle(random_index_list)

    output_index_list = []
    idx_counter = 0
    while len(output_index_list) < sample_amount:
        index = random.randint(0, int(math.ceil(weight_sum)))
        # index = random_index_list[idx_counter]
        # stop at the first true
        selected_idx = np.argmax(cdf > index)
        # print("index", index)
        # print("selected_idx", selected_idx)

        if selected_idx not in output_index_list:
            output_index_list.append(selected_idx)
        idx_counter += 1

    # output
    output_index_list = np.array(output_index_list)
    output = input_data[output_index_list]
    print(output.shape)
    np.save(sampled_file, output)
    
    
    # sum over the whole input_data's weight by [speed*density]
    pdf = []
    for _, v in enumerate(output):
        delta_speed = np.amax(v[:, :, 2]) - np.amin(v[:, :, 2])
        mean_flow = np.mean(v[:, :, 0])
        importance_weight = delta_speed * mean_flow
        pdf.append(importance_weight)
    pdf = np.array(pdf)

    plot.figure(2)
    plot.hist(pdf, bins=10, normed=is_PDF)

    plot.savefig(sampled_dist_plot)
    plot.show()


if __name__ == '__main__':
    is_sampling()
