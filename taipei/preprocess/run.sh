#!/bin/sh


echo "generate_base_som_1d 0"
python generate_base_som_1d.py fix_raw_data_time0.json
echo "training 0"
python ../2dcnn_multi_gpu/train_2dcnn.py
echo "generate_base_som_1d 1"
python generate_base_som_1d.py fix_raw_data_time1.json
echo "training 1"
python ../2dcnn_multi_gpu/train_2dcnn.py