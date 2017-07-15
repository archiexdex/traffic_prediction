#!/bin/sh

# new_path='backlog_new/no_over_data_mile_15_28.5_total_60_predict_1_5/test_log_'
# loss_path='backlog_loss/no_over_data_mile_15_28.5_total_60_predict_1_5/test_log_'

# path_num=0
# day=698


# while [ "${day}" != "703" ]
# do
#     python test_conv.py --log_dir=$new_path$path_num"/" --day=$day
#     # python3 B_test_lstm.py --log_dir=$loss_path$path_num"/" --day=$day

#     # mv $loss_path$path_num"/predicted_loss" $new_path$path_num"/"
#     path_num=$(($path_num+1))
#     day=$(($day+1))
# done 


raw_data="batch_no_over_data_mile_15_28.5_total_"
label_data="label_no_over_data_mile_15_28.5_total_"

dir="predict_1_"

predict_time=15

while [ "${predict_time}" != "25" ]
do
    T=`expr $predict_time \* 12`

    python train_conv.py --raw_data=$raw_data$T"_"$dir$predict_time".npy" --label_data=$label_data$T"_"$dir$predict_time".npy" \
    --checkpoints_dir='backlog_new/'$dir$predict_time'/checkpoints/' --log_dir='backlog_new/'$dir$predict_time'/log/'
    predict_time=$(($predict_time+5))
done