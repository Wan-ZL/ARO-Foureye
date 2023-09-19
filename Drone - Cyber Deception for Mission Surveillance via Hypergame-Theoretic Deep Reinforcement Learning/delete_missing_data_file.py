'''
Project     ：Drone-DRL-HT 
File        ：delete_missing_data_file.py
Author      ：Zelin Wan
Date        ：6/2/23
Description : 
'''

import os
import shutil

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from multiprocessing import Process

# Mac path
dir_path = './data/150_5_5/'
# Server path
# dir_path = '/projects/zelin1/Drone_DRL_HT/data/150_5_0/'

def_select_method_list = ['fixed', 'HEU', 'DRL', 'HT-DRL', 'No-Defense', 'IDS', 'CD']
att_select_method_list = ['fixed', 'HEU', 'DRL', 'HT-DRL']

# scheme_list = ['scheme_DefAtt']
# HEU_usage_list = ['HEU_def=False_att=False', 'HEU_def=True_att=False', 'HEU_def=False_att=True', 'HEU_def=True_att=True']
tag = "Accumulated Reward/Attacker"

# find file that has not enough data
required_data_size = 600
delete_data_size = 500

def show_missing_data_file(path):
    file_list = os.listdir(path)
    # print("file_list: ", file_list)
    for file in file_list:
        # ignore .DS_Store
        if file == '.DS_Store':
            continue

        file_path = path + '/' + file
        event_acc = EventAccumulator(file_path)
        event_acc.Reload()
        # print("file_path: ", file_path)
        try:
            file_size = len(event_acc.Scalars(tag))
        except:
            print("ERROR File!!! Delete:", file_path)
            shutil.rmtree(file_path)
            continue
        # print("file size:", file_path, file_size)
        if file_size < required_data_size:
            # delete file
            if file_size < delete_data_size:
                print("Delete:", file_path, file_size)
                shutil.rmtree(file_path)
            else:
                print("file size:", file_path, file_size)


if __name__ == '__main__':
    process_list = []
    for def_select_method in def_select_method_list:
        for att_select_method in att_select_method_list:
            path = dir_path + 'scheme_' + att_select_method + '-' + def_select_method
            print("path: ", path)
            # parallel
            p = Process(target=show_missing_data_file, args=(path,))
            process_list.append(p)
            p.start()
    # for entropy_threshold in entropy_threshold_list:
    #     for epsilon in epsilon_list:
    #         for scheme in scheme_list:
    #             for HEU_usage in HEU_usage_list:
    #                 path = dir_path + 'entropy_threshold_'+str(entropy_threshold)+'_epsilon_'+str(epsilon)+ '/' + scheme + '/' + HEU_usage
    #                 print("path: ", path)
    #                 # parallel
    #                 p = Process(target=show_missing_data_file, args=(path,))
    #                 process_list.append(p)
    #                 p.start()




