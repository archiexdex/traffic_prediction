import numpy as np

# Parameters
predict_time = 20
num_steps = 12 
time_preried = predict_time * num_steps

st = 15
ed = 28.5

start_predict = 1

is_over = 0

root_path = "/home/nctucgv/Documents/TrafficVis_Run/src/traffic_flow_detection/"
data_path = "/home/nctucgv/Documents/TrafficVis_Run/"


def read_file(filename, vec, week_list, time_list, week, st, ed):
    """
    Param:
        filename:
        vec:
        week_list:
        time_list:
        week:
        st:
        ed:
    """

    # 1. Get real file path
    filename = data_path + "VD_data/mile_base/" + filename

    # 2. Open file as binary file 
    with open(filename, "rb") as binaryfile:
        binaryfile.seek(0)
        ptr = binaryfile.read(4)

        data_per_day = 1440
        VD_size = int.from_bytes(ptr, byteorder='little')
        ptr = binaryfile.read(4)
        day_max = int.from_bytes(ptr, byteorder='little')

        # initialize list
        dis = int((ed - st) * 2 + 1)
        vt = len(vec)
        wt = len(week_list)
        tt = len(time_list)
        for i in range(day_max):
            vec.append([0] * dis)
            week_list.append([0] * dis)
            time_list.append([0] * dis)

        index = 0
        for i in range(VD_size):

            if st <= i / 2 and i / 2 <= ed:
                for j in range(day_max):
                    ptr = binaryfile.read(2)
                    tmp = int.from_bytes(ptr, byteorder='little')
                    vec[vt + j][index] = tmp
                    week_list[wt +j][index] = (week + int(j / data_per_day)) % 7
                    time_list[tt + j][index] = j % data_per_day
                index = index + 1
            elif ed < i / 2:
                break
            else:
                binaryfile.read(2)


raw_data = []


try:
     raw_data = np.load(root_path+"fix_raw_data_"+str(st)+"_"+str(ed)+".npy")
except:
    pass

if len(raw_data) == 0:

    try:
        raw_data = np.load(root_path+"raw_data.npy")
    except:
        pass

    if len(raw_data) == 0:
        
        print("raw_data is not exist. QAQ")
        # Initialize lists
        density_list = []
        flow_list = []
        speed_list = []
        week_list = []
        time_list = []

        # Read files
        print("Reading 2012...")
        read_file("density_N5_N_2012_1_12.bin", density_list, [], [], 0, st, ed)
        read_file("flow_N5_N_2012_1_12.bin"   , flow_list, [], [], 0, st, ed)
        read_file("speed_N5_N_2012_1_12.bin", speed_list, week_list, time_list, 0, st, ed)

        print("Reading 2013...")
        read_file("density_N5_N_2013_1_12.bin", density_list, [], [], 2, st, ed)
        read_file("flow_N5_N_2013_1_12.bin"   , flow_list, [], [], 2, st, ed)
        read_file("speed_N5_N_2013_1_12.bin", speed_list, week_list, time_list, 2, st, ed)

        print("Reading 2014...")
        read_file("density_N5_N_2014_1_12.bin", density_list, [], [], 3, st, ed)
        read_file("flow_N5_N_2014_1_12.bin"   , flow_list, [], [], 3, st, ed)
        read_file("speed_N5_N_2014_1_12.bin", speed_list, week_list, time_list, 3, st, ed)

        # fix data
        # data[i][10] are always 0 and data[i][13] in 2012 are always 0
        print("Fixing data...")
        for i in range(len(speed_list)):
            if density_list[i][10] == 0:
                density_list[i][10] = int((density_list[i][9] + density_list[i][11]) / 2)
            if density_list[i][13] == 0:
                density_list[i][13] = int((density_list[i][12] + density_list[i][14]) / 2)
            if flow_list[i][10] == 0:
                flow_list[i][10] = int((flow_list[i][9] + flow_list[i][11]) / 2) 
            if flow_list[i][13] == 0:
                flow_list[i][13] = int((flow_list[i][12] + flow_list[i][14]) / 2)
            if speed_list[i][10] == 0:
                speed_list[i][10] = int((speed_list[i][9] + speed_list[i][11]) / 2)
            if speed_list[i][13] == 0:
                speed_list[i][13] = int((speed_list[i][12] + speed_list[i][14]) / 2)

        # merge different dimention data in one
        raw_data = np.stack((density_list, flow_list, speed_list, week_list, time_list), axis=2)

        # delete data
        del density_list
        del flow_list
        del speed_list
        del week_list
        del time_list

        # save raw data
        np.save("raw_data", raw_data)

    else:
        print("raw_data is exist. ^ _ ^")
    
    c = 0
    p = []
    b = []
    print("Removing illegal data...")

    for i in raw_data:
        t = []
        flg = False
        
        for k in i:
            t = np.argwhere( (k[0] == 0 or 100 < k[0]) or (k[1] == 0 or 40 * 2 < k[1]) or (k[2] == 0 or 120 < k[2]) )
            if len(t) > 0:
                flg = True
                break
        if flg:
            b.append(c)
        if b.count > 29:
            for i in b:
                p.append(i)
            b = []
        print(c)
        c += 1
    
    if b.count > 29:
        for i in b:
            p.append(i)
        b = []

    for i in p:
        raw_data[i] = [-1,-1,-1,-1,-1]
    np.save("fix_raw_data",raw_data)

    del b
    del p
    del c

else:
    print("fix_data is exist. ^ _ ^")


print("start to distribute data...")

batch_data = []
label_data = []

i = 0
# for i in range(len(raw_data) - time_preried - predict_time):
while i < len(raw_data) - time_preried - predict_time:
    tmp = raw_data[i:i+time_preried]
    ret = []
    is_good = 0
    # for training data
    j = 0
    while j < len(tmp):
        a = tmp[j:j+predict_time]
        sump = 0
        flg = 0
        for k in a:
            if k[0][0] != -1:
                sump += k
                flg += 1
        if flg <= predict_time - 2:
            is_good += 1
            break
        sump = sump / flg
        ret.append(sump)
                
        j += predict_time

    if is_good > 0:
        i += 1
        continue

    # for label data
    tmp = raw_data[i+time_preried+start_predict-1:i+time_preried+start_predict+predict_time-1]
    ret1 = []

    j = 0
    while j < len(tmp):
        a = tmp[j:j+predict_time]
        sump = 0
        flg = 0
        for k in a:
            if k[0][0] != -1:
                sump += k
                flg += 1
        if flg <= predict_time - 2:
            is_good += 1
            break
        sump = sump / flg
        ret1.append(sump)
                
        j += predict_time

    if is_good > 0:
        i += 1
        continue
    else:
        batch_data.append(ret)
        label_data.append(ret1[0])
        i += predict_time
    
    print(i)


print(len(batch_data))

if is_over == 0:
    np.save(root_path+"batch_no_over_data_mile_"+str(st)+"_"+str(ed)+"_total_"+str(time_preried)+"_predict_"+str(start_predict)+"_"+str(start_predict+predict_time-1), batch_data)
    np.save(root_path+"label_no_over_data_mile_"+str(st)+"_"+str(ed)+"_total_"+str(time_preried)+"_predict_"+str(start_predict)+"_"+str(start_predict+predict_time-1), label_data)    

else:
    np.save(root_path+"batch_data_mile_"+str(st)+"_"+str(ed)+"_total_"+str(time_preried)+"_predict_"+str(start_predict)+"_"+str(start_predict+predict_time-1), batch_data)
    np.save(root_path+"label_data_mile_"+str(st)+"_"+str(ed)+"_total_"+str(time_preried)+"_predict_"+str(start_predict)+"_"+str(start_predict+predict_time-1), label_data)    


print("Finish")
