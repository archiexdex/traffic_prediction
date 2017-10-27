import os
import numpy as np
import json


LOAD_PATH = "/home/xdex/Desktop/traffic_flow_detection/taipei/training_data/current_data/lane_base/"
SAVE_PATH = "/home/xdex/Desktop/traffic_flow_detection/taipei/training_data/current_data/gruop_base/"

ROOT_PATH = "/home/xdex/Desktop/traffic_flow_detection/taipei/training_data/current_data/"

def main():
    vd_grp_lane_dict = {}
    with open( os.path.join(ROOT_PATH, "vd_grp_lane.json") ) as fp:
        vd_grp_lane_dict = json.load(fp)
    

    for vd in vd_grp_lane_dict:
        for grp in vd_grp_lane_dict[vd]:
            for lane in vd_grp_lane_dict[vd][grp]:
                file_name = vd + "_" + lane + ".npy"
                if not os.path.exists( os.path.join(LOAD_PATH, file_name) )
                    print(file_name " doesn't exist! QQ")
            
                

    pass

if __name__ == "__main__":
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)
    main()
