'''
Project     ：Drone-DRL-HT 
File        ：Gym_Defender_and_Attacker.py
Author      ：Zelin Wan
Date        ：2/7/23
Description : 
'''


'''
Project     ：gym-drones
File        ：Gym_HoneyDrone_Defender_and_Attacker.py
Author      ：Zelin Wan
Date        ：6/25/22
Description : This is gym-based environment. It takes two actions (one from defender, one from attacker), and return
two rewards, two observations (one for defender, one for attacker).
'''

from gym import Env
from gym.spaces import Discrete, Box
import gym
import math
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# import torchvision.transforms as T

from multiprocessing import cpu_count
from multiprocessing import Process

import time
import argparse
import gym

import random
import numpy as np

from model_System import system_model
from model_Defender import defender_model
from model_Attacker import attacker_model
from typing import TypeVar, Tuple
ObsType = TypeVar("ObsType")


class HyperGameSim(Env):
    def __init__(self, fixed_seed=False, miss_dur=150, target_size=5, max_att_budget=5, num_HD=5, def_select_method='fixed'):
        # variable for bullet drone env
        self.print = False
        self.gui = False
        self.system = None
        self.defender = None
        self.attacker = None
        self.frameN = 1
        self.TARGET_POS = None
        self.ctrl = None
        self.CTRL_EVERY_N_STEPS = None
        self.env = None
        self.pybullet_action = None
        self.START = None
        self.ARGS = None
        self.start_time = None
        self.initDroneEnv(miss_dur, target_size, max_att_budget, num_HD, def_select_method)

        # variable for current env
        self.fixed_seed = fixed_seed
        self.miss_dur = miss_dur
        self.target_size = target_size
        self.max_att_budget = max_att_budget
        self.num_HD = num_HD
        self.def_select_method = def_select_method
        self.action_space = dict()
        self.action_space['def'] = Discrete(self.defender.number_of_strategy)
        self.action_space['att'] = Discrete(self.attacker.number_of_strategy)
        self.observation_space = dict()
        # Defender
        temp_low_array_def = [] # [0., 0., 0., 0.]
        temp_high_array_def = [] # [self.defender.system.mission_duration_max, 1., self.attacker.max_att_budget, self.attacker.max_att_budget]
        # add duration
        # temp_low_array_def.append(0.)
        # temp_high_array_def.append(self.defender.system.mission_duration_max)
        # add ratio of mission completion
        temp_low_array_def.append(0.)
        temp_high_array_def.append(1.)
        # add number of success attack
        # temp_low_array_def.append(0.)
        # temp_high_array_def.append(self.attacker.max_att_budget)
        # add number of attack
        # temp_low_array_def.append(0.)
        # temp_high_array_def.append(self.attacker.max_att_budget)
        # add drone position
        # for _ in range(self.system.num_MD + self.system.num_HD + 1):
        #     temp_low_array_def += [0., 0., 0.]
        # for _ in range(self.system.num_MD + self.system.num_HD + 1):
        #     temp_high_array_def += [self.system.map_size, self.system.map_size, self.system.map_size]
        # add scan map
        temp_low_array_def += [0]*25
        temp_high_array_def += [self.system.min_scan_requirement]*25
        # add HEU
        # temp_array = temp_array + [0., 0., 0., 0., 0.]
        # add HEU
        # temp_array += [1, 1, 1, 1, 1]
        # set observation space for defender
        low_array_def = np.array(temp_low_array_def)
        high_array_def = np.array(temp_high_array_def)
        self.observation_space['def'] = Box(low=low_array_def, high=high_array_def)
        # Attacker
        temp_low_array_att = []
        temp_high_array_att = []
        # add duration
        # temp_low_array_att.append(0.)
        # temp_high_array_att.append(self.defender.system.mission_duration_max)
        # add number of success attack
        # temp_low_array_att.append(0.)
        # temp_high_array_att.append(self.attacker.max_att_budget)
        # add number of attack
        # temp_low_array_att.append(0.)
        # temp_high_array_att.append(self.attacker.max_att_budget)
        # add drone signal
        # temp_low_array_att += [self.attacker.undetect_dbm for _ in range(self.system.num_MD+self.system.num_HD)]
        # temp_high_array_att += [20. for _ in range(self.system.num_MD+self.system.num_HD)]
        # add drone number in each strategy
        temp_low_array_att += [0] * self.attacker.number_of_strategy
        temp_high_array_att += [self.system.num_MD + self.system.num_HD] * self.attacker.number_of_strategy
        # set observation space for attacker
        low_array_att = np.array(temp_low_array_att)
        high_array_att = np.array(temp_high_array_att)
        self.observation_space['att'] = Box(low=low_array_att, high=high_array_att)
        self.step_count = 0
        self.pre_scan_ratio = 0.0


    def set_random_seed(self):
        # reset seed for new episode
        if self.fixed_seed:
            np.random.seed(0)
            random.seed(0)


    def reset(self):
        '''

        Args:
            *args:
            miss_dur:  mission_duration_max. Vary this for sensitivity analysis.

        Returns:

        '''
        self.start_time = time.time()
        self.initDroneEnv(self.miss_dur, self.target_size, self.max_att_budget, self.num_HD, self.def_select_method)
        # self.system.mission_duration_max = miss_dur
        # self.system.map_cell_number = target_size

        self.set_random_seed()
        step_count = 0
        self.pre_scan_ratio = 0.0

        return self.get_state()

    def get_state(self):
        # def
        # drone_pos_set = np.array([])
        # for drone in self.system.Drone_set_include_creashed:
        #     drone_pos_set = np.append(drone_pos_set, drone.xyz_temp)

        temp_state_def = []
        # add duration
        # temp_state_def.append(self.system.mission_duration)
        # add ratio of mission completion
        temp_state_def.append(self.system.scanCompletePercent())
        # # add number of success attack
        # temp_state_def.append(self.attacker.att_succ_counter_one_round)
        # # add number of attack
        # temp_state_def.append(self.attacker.att_counter_one_round)
        # add drone position
        # def_pos_set = np.array([])
        # for drone in self.system.Drone_set_include_creashed:
        #     def_pos_set = np.append(def_pos_set, drone.xyz_temp)
        # temp_state_def = np.append(temp_state_def, def_pos_set)
        # add scan map
        temp_state_def = np.append(temp_state_def, self.system.scan_cell_map.flatten())
        # convert to defender's observation
        state_def = np.array(temp_state_def)
        # state_def = np.append(state_def, def_pos_set)


        # att
        # attack success ratio
        # if self.attacker.att_counter == 0:
        #     att_succ_rate = 0
        # else:
        #     att_succ_rate = self.attacker.att_succ_counter / self.attacker.att_counter
        # set of attacker's received signal
        att_signal_set = []
        # add duration
        # att_signal_set.append(self.system.mission_duration)
        # add number of success attack
        # att_signal_set.append(self.attacker.att_succ_counter_one_round)
        # add number of attack
        # att_signal_set.append(self.attacker.att_counter_one_round)
        # add drone signal
        # for id in range(self.system.num_MD + self.system.num_HD):
        #     if id in self.attacker.obs_sig_dict:
        #         att_signal_set.append(self.attacker.obs_sig_dict[id])
        #     else:
        #         att_signal_set.append(self.attacker.undetect_dbm)
        # add drone number in each strategy
        for id in range(self.attacker.number_of_strategy):
            att_signal_set.append(len(self.attacker.S_target_dict[id]))
        # convert to attacker's observation
        state_att = np.array(att_signal_set)
        # state_att = [self.system.mission_duration, att_succ_rate] + att_signal_set
        # state_att = np.array([self.system.mission_duration, self.attacker.att_succ_counter_one_round,
        #                       self.attacker.att_counter_one_round])
        # combine
        state = {}
        state['def'] = state_def
        state['att'] = state_att

        return state


    def initDroneEnv(self, miss_dur, target_size, max_att_budget, num_HD, def_select_method):
        # create model class
        self.system = system_model(mission_duration_max=miss_dur, map_cell_number=target_size, num_HD=num_HD)
        self.defender = defender_model(self.system)
        self.attacker = attacker_model(self.system, max_att_budget, def_select_method)



    # step inherent from gym.step, but changed the return format (two rewards)
    def step(self, action_def=None, action_att=None) -> Tuple[dict, dict, bool, dict]:
        '''
        Args:
            action_def: defender's action (if no given, random action will be applied)
            action_att: attacker's action (if no given, random action will be applied)

        Returns: state['def']+state['att'], reward['def']+reward['att'], done, info
        '''
        self.step_count += 1

        if action_def is None:
            # If no action is given, random action from 0 to 9 will be applied
            # action_def = np.random.randint(0, 10, dtype=np.int64)
            action_def = np.int64(4) #np.int64(4) # np.int64(5)
        if action_att is None:
            # If no action is given, random action from 0 to 9 will be applied
            # action_att = np.random.randint(0, 10, dtype=np.int64)
            action_att = np.int64(4) # np.int64(5)

        # pybullet environment
        self.roundBegin(action_def, action_att)
        self.moveDrones()

        # hyperGame environment state
        state = self.get_state()

        if self.system.is_mission_Not_end():
            done = False
        else:
            done = True
            print("--- game duration: %s seconds ---" % round(time.time() - self.start_time, 1))

        # Defender's Reward
        energy_HD = self.system.HD_one_round_consume()
        energy_MD = self.system.MD_one_round_consume()
        N_AC = len(self.system.MD_connected_set) + len(self.system.HD_connected_set)
        # reward_def = math.exp(-1/N_AC) if N_AC else 0
        # scan_ratio = self.system.scanCellMapSum()
        # scan_ratio = self.system.scanCompletePercent()
        scan_ratio = self.system.scanCompleteCount()
        reward_def = scan_ratio - self.pre_scan_ratio
        self.pre_scan_ratio = scan_ratio
        # reward_def = -0.1
        # if done:
        #     if self.system.mission_success:
        #         reward_def = 100
        # reward_def = 1.0

        # Attacker's Reward
        one_round_att_counter = float(self.attacker.att_counter_one_round)
        one_round_att_succ_counter = float(self.attacker.att_succ_counter_one_round)
        reward_numerate = float(1+(self.system.num_MD + self.system.num_HD-self.system.aliveDroneCount()))
        reward_demonirate = float(self.system.mission_duration)
        # reward_att = 1 - math.exp(-(reward_numerate/reward_demonirate))
        reward_att = one_round_att_succ_counter


        # if self.attacker.att_counter:
        #     one_round_att_counter = float(self.attacker.att_counter)
        #     one_round_att_succ_counter = float(self.attacker.att_succ_counter)
        #     reward_att = w_AS * (one_round_att_succ_counter/one_round_att_counter) - w_AC * (one_round_att_counter/self.attacker.max_att_budget)
        #
        # else:
        #     # the self.attacker.att_counter=0 lead to whole reward=0
        #     reward_att = 0

        reward = {}
        reward['def'] = reward_def
        # reward['att'] = reward_att
        # reward['att'] = 0 - reward_def # TODO: this is for testing (zero sum reward)
        reward['att'] = self.system.map_cell_number**2 - self.system.scanCompleteCount()




        info = {}
        # info['att_succ_rate'] = att_succ_rate
        info['action_def'] = action_def
        info['action_att'] = action_att
        info["mission_condition"] = self.system.mission_condition
        total_energy_consump = 0 # energy consumption of all drones
        for HD in self.system.HD_set:
            total_energy_consump += HD.accumulated_consumption
        for MD in self.system.MD_set:
            total_energy_consump += MD.accumulated_consumption
        info["total_energy_consump"] = total_energy_consump
        info["scan_percent"] = self.system.scanCompletePercent()
        info["map_cell_number"] = self.system.map_cell_number
        info["MD_active_num"] = len(self.system.MD_set)
        info["MD_connected_num"] = len(self.system.MD_connected_set)
        info["MDHD_active_num"] = len(self.system.MD_set) + len(self.system.HD_set)
        info["MDHD_connected_num"] = len(self.system.MD_connected_set) + len(self.system.HD_connected_set)
        info["spent_time"] = self.system.mission_duration
        info["remaining_time"] = self.system.mission_duration_max - self.system.mission_duration
        info["remaining_time_ratio"] = 1 - (self.system.mission_duration / self.system.mission_duration_max)
        info['mission_duration_max'] = self.system.mission_duration_max
        info["energy_HD"] = energy_HD
        info["energy_MD"] = energy_MD
        return state, reward, done, info

    def roundBegin(self, action_def, action_att):
        # check if mission complete
        self.system.check_mission_complete()

        # path planning for MD
        if self.defender.is_calc_trajectory():
            MD_trajectory = self.defender.MD_trajectory

        # Drones state update
        self.system.Drone_state_update()

        # show real time scan
        if self.print: print("map \n", self.system.scan_cell_map)

        # # for MD update next destination
        # if self.frameN <= self.update_freq:  # first round doesn't check scan_map
        #     self.defender.update_MD_next_destination_no_scanMap()
        # else:
        #     self.defender.update_MD_next_destination()
        self.defender.update_MD_next_destination()

        # # avoid HD crash when creating
        # if self.frameN <= self.update_freq:
        #     for HD in self.system.HD_set:
        #         HD.assign_destination(self.defender.HD_locations[HD.ID])
        for HD in self.system.HD_set:
            HD.assign_destination(self.defender.HD_locations[HD.ID])

        # # for HD update next destination
        self.defender.update_HD_next_destination()

        # for RLD update the fixed destination
        self.defender.update_RLD_next_destination()

        # Attacker
        self.attacker.observe()
        self.attacker.select_strategy(action_att)
        self.attacker.action()

        # Defender
        self.defender.select_strategy(action_def)
        self.defender.action()

        # Update Battery Consumption Rate
        def_stra = self.defender.strategy
        for MD in self.system.MD_set:
            MD.consume_rate_update(def_stra)
        for HD in self.system.HD_set:
            HD.consume_rate_update(def_stra)

    def moveDrones(self):
        self.updateTempLoca(self.system.MD_set)
        self.updateTempLoca(self.system.HD_set)
        self.updateTempLoca(self.system.RLD_set)

        # scan count, and check if crashed
        self.system.MD_environment_interaction()
        self.system.HD_environment_interaction()

        # energy consumption of MD and HD
        self.system.battery_consume()

        # update Drone's neighbor table
        self.system.update_neighbor_table()
        # update Drone's fleet table
        self.system.update_RLD_connection()
        return

    def render(self, *args):
        pass



    def updateTempLoca(self, drone_set):
        for drone in drone_set:
            drone.xyz_temp = drone.xyz_destination
            # if drone.crashed:
            #     continue
            # vector_per_frame = (drone.xyz_destination - drone.xyz_temp) / (
            #         2 * control_freq_hz)  # multiple 2 to slow down drone speed for a long distance travel
            #
            # # avoid too fast movement. It cause crash
            # vector_magnitude = np.linalg.norm(vector_per_frame)
            # if vector_magnitude > drone.speed_per_frame_max:
            #     vector_per_frame = vector_per_frame / (vector_magnitude / drone.speed_per_frame_max)
            # drone.xyz_temp = vector_per_frame + drone.xyz_temp

    def drones_distance(self, drone_x_location, drone_y_location):
        distance_squre = np.square(drone_x_location - drone_y_location)
        return np.sqrt(np.sum(distance_squre))
