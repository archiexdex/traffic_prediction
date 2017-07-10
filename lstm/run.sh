#!/bin/sh

new_path='backlog_new/no_over_data_mile_15_28.5_total_60_predict_1_5/test_log_'
loss_path='backlog_loss/no_over_data_mile_15_28.5_total_60_predict_1_5/test_log_'

path_num=0
day=150


while [ "${day}" != "158" ]
do
    python3 test_lstm.py --log_dir=$new_path$path_num"/" --day=$day
    python3 B_test_lstm.py --log_dir=$loss_path$path_num"/" --day=$day

    mv $loss_path$path_num"/predicted_loss" $new_path$path_num"/"
    path_num=$(($path_num+1))
    day=$(($day+1))
done 
