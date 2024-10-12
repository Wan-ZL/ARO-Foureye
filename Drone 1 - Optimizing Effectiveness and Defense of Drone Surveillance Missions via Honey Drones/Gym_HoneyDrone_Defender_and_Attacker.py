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
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# import torchvision.transforms as T

import time
import argparse
import pybullet as p
import gym

import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

from model_System import system_model
from model_Defender import defender_model
from model_Attacker import attacker_model
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.utils import sync, str2bool
from typing import TypeVar, Tuple
ObsType = TypeVar("ObsType")


class HyperGameSim(Env):
    def __init__(self, fixed_seed=True, miss_dur=30, target_size=5, max_att_budget=5, num_HD=2, defense_strategy=0):
        # variable for bullet drone env
        self.print = False
        self.gui = False
        self.system = None
        self.defender = None
        self.attacker = None
        self.frameN = 1
        self.update_freq = 500
        self.TARGET_POS = None
        self.ctrl = None
        self.CTRL_EVERY_N_STEPS = None
        self.env = None
        self.pybullet_action = None
        self.START = None
        self.ARGS = None
        self.start_time = None
        self.initDroneEnv(miss_dur, target_size, max_att_budget, num_HD, defense_strategy)
        self.close_env()  # close client in init for avoiding client limit error

        # variable for current env
        self.fixed_seed = fixed_seed
        self.miss_dur = miss_dur
        self.target_size = target_size
        self.max_att_budget = max_att_budget
        self.num_HD = num_HD
        self.defense_strategy = defense_strategy
        self.action_space = dict()
        self.action_space['def'] = Discrete(self.defender.number_of_strategy)
        self.action_space['att'] = Discrete(self.attacker.number_of_strategy)
        self.observation_space = dict()
        # [time duration, ratio of mission complete]
        self.observation_space['def'] = Box(low=np.array([0., 0.]),
                                     high=np.array([self.defender.system.mission_duration_max, 1.]))
        # [time duration, ratio of attack success, [set of received signal level for all drones]]
        low_array = np.array([0., 0., 0.]+[self.attacker.undetect_dbm for _ in range(self.system.num_MD+self.system.num_HD)])    # 'undetect_dbm' dBm means the signal is uundetectable
        # high_array = np.array([self.defender.system.mission_duration_max, 1.]+[20. for _ in range(self.system.num_MD+self.system.num_HD)])
        high_array = np.array([self.defender.system.mission_duration_max, self.attacker.max_att_budget, self.attacker.max_att_budget] + [20. for _ in range(self.system.num_MD + self.system.num_HD)])
        self.observation_space['att'] = Box(low=low_array, high=high_array)


    def set_random_seed(self):
        # reset seed for new episode
        if self.fixed_seed:
            np.random.seed(0)
            random.seed(0)

    # close client. This function check the connection status before close it. Use this function to avoid client limit.
    def close_env(self):
        if self.env is not None:
            if p.getConnectionInfo(self.env.getPyBulletClient())['isConnected']:
                # print("closing client", self.env.getPyBulletClient())
                self.env.close()

    def reset(self, *args, miss_dur=30, target_size=5, max_att_budget=5, num_HD=2):
        '''

        Args:
            *args:
            miss_dur:  mission_duration_max. Vary this for sensitivity analysis.

        Returns:

        '''
        self.start_time = time.time()
        self.initDroneEnv(miss_dur, target_size, max_att_budget, num_HD, self.defense_strategy)
        # self.system.mission_duration_max = miss_dur
        # self.system.map_cell_number = target_size

        self.set_random_seed()

        # def
        # mission_complete_ratio = self.system.scanCompletePercent()
        state_def = [self.system.mission_duration, self.system.scanCompletePercent()]
        # att
        # attack success ratio
        # if self.attacker.att_counter == 0:
        #     att_succ_rate = 0
        # else:
        #     att_succ_rate = self.attacker.att_succ_counter / self.attacker.att_counter
        # set of attacker's received signal
        att_signal_set = []
        for id in range(self.system.num_MD + self.system.num_HD):
            if id in self.attacker.obs_sig_dict:
                att_signal_set.append(self.attacker.obs_sig_dict[id])
            else:
                att_signal_set.append(self.attacker.undetect_dbm)

        # state_att = [self.system.mission_duration, att_succ_rate] + att_signal_set
        state_att = [self.system.mission_duration, self.attacker.att_succ_counter, self.attacker.att_counter] + att_signal_set
        # combine
        state = {}
        state['def'] = np.array(state_def)
        state['att'] = np.array(state_att)

        return state

    def initDroneEnv(self, miss_dur=30, target_size=5, max_att_budget=5, num_HD=2, defense_strategy=0):
        # create model class
        self.system = system_model(mission_duration_max=miss_dur, map_cell_number=target_size, num_HD=num_HD)
        self.defender = defender_model(self.system)
        self.attacker = attacker_model(self.system, max_att_budget, defense_strategy)

        if self.print: print("attacker locaiton", self.attacker.xyz)


        #### default parameters:
        num_MD = self.system.num_MD  # number of MD (in index, MD first then HD)
        num_HD = self.system.num_HD  # number of HD

        #### Define and parse (optional) arguments for the script ##
        parser = argparse.ArgumentParser(
            description='Helix flight script using CtrlAviary or VisionAviary and DSLPIDControl')
        parser.add_argument('--drone', default="cf2p", type=DroneModel, help='Drone model (default: CF2X)', metavar='',
                            choices=DroneModel)
        parser.add_argument('--num_drones', default=num_HD + num_MD + 1, type=int, help='Number of drones', metavar='')
        parser.add_argument('--physics', default="pyb", type=Physics, help='Physics updates (default: PYB)', metavar='',
                            choices=Physics)
        parser.add_argument('--vision', default=False, type=str2bool,
                            help='Whether to use VisionAviary (default: False)', metavar='')
        parser.add_argument('--gui', default=self.gui, type=str2bool,
                            help='Whether to use PyBullet GUI (default: True)',
                            metavar='')
        parser.add_argument('--record_video', default=False, type=str2bool,
                            help='Whether to record a video (default: False)', metavar='')
        parser.add_argument('--plot', default=True, type=str2bool,
                            help='Whether to plot the simulation results (default: True)', metavar='')
        parser.add_argument('--user_debug_gui', default=False, type=str2bool,
                            help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
        parser.add_argument('--aggregate', default=True, type=str2bool,
                            help='Whether to aggregate physics steps (default: True)', metavar='')
        parser.add_argument('--obstacles', default=False, type=str2bool,
                            help='Whether to add obstacles to the environment (default: False)', metavar='')
        parser.add_argument('--simulation_freq_hz', default=30, type=int,
                            help='Simulation frequency in Hz (default: 240)', metavar='')
        parser.add_argument('--control_freq_hz', default=20, type=int, help='Control frequency in Hz (default: 48)',
                            metavar='')
        parser.add_argument('--duration_sec', default=5, type=int,
                            help='Duration of the simulation in seconds (default: 5)', metavar='')
        self.ARGS = parser.parse_args()

        #### Initialize the simulation #############################
        H = 1
        H_STEP = .05
        R = .3

        INIT_XYZS = np.array([[R * np.cos((i / 6) * 2 * np.pi + np.pi / 2),
                               R * np.sin((i / 6) * 2 * np.pi + np.pi / 2) - R, H + i * H_STEP] for i in
                              range(self.ARGS.num_drones)])

        INIT_RPYS = np.array([[0, 0, i * (np.pi / 2) / self.ARGS.num_drones] for i in range(self.ARGS.num_drones)])
        AGGR_PHY_STEPS = int(self.ARGS.simulation_freq_hz / self.ARGS.control_freq_hz) if self.ARGS.aggregate else 1

        # gym-like environment
        self.close_env()  # close previous client
        self.env = CtrlAviary(drone_model=self.ARGS.drone,
                              num_drones=self.ARGS.num_drones,
                              initial_xyzs=INIT_XYZS,
                              initial_rpys=INIT_RPYS,
                              physics=self.ARGS.physics,
                              neighbourhood_radius=10,
                              freq=self.ARGS.simulation_freq_hz,
                              aggregate_phy_steps=AGGR_PHY_STEPS,
                              gui=self.ARGS.gui,
                              record=self.ARGS.record_video,
                              obstacles=self.ARGS.obstacles,
                              user_debug_gui=self.ARGS.user_debug_gui,
                              print_result = False
                              )
        # print("creating client", self.env.getPyBulletClient())

        PYB_CLIENT = self.env.getPyBulletClient()

        #### Initialize the controllers ############################
        self.ctrl = [DSLPIDControl(drone_model=self.ARGS.drone) for i in range(self.ARGS.num_drones)]

        #### Run the simulation ####################################
        #### (0,0) is base station ####
        self.CTRL_EVERY_N_STEPS = int(np.floor(self.env.SIM_FREQ / self.ARGS.control_freq_hz))
        if self.print: print("CTRL_EVERY_N_STEPS", self.CTRL_EVERY_N_STEPS)
        self.pybullet_action = {str(i): np.array([12713, 12713, 12713, 12713]) for i in range(self.ARGS.num_drones)}
        self.START = time.time()
        self.frameN = 1

        # initial position for drones (MD+HD)
        for MD in self.system.MD_set:
            MD.assign_destination(INIT_XYZS[MD.ID])
            MD.xyz_temp = MD.xyz
        for HD in self.system.HD_set:
            HD.assign_destination(INIT_XYZS[HD.ID])
            HD.xyz_temp = HD.xyz
        for RLD in self.system.RLD_set:
            RLD.assign_destination((INIT_XYZS[RLD.ID]))
            RLD.xyz_temp = RLD.xyz

        # disconnect exist physics client, and create a new clident
        p.loadURDF("duck_vhacd.urdf", self.attacker.xyz, physicsClientId=PYB_CLIENT)

        self.update_freq = self.system.update_freq



    # step inherent from gym.step, but changed the return format (two rewards)
    def step(self, action_def=None, action_att=None) -> Tuple[dict, dict, bool, dict]:
        '''
        Args:
            action_def: defender's action (if no given, random action will be applied)
            action_att: attacker's action (if no given, random action will be applied)

        Returns: state['def']+state['att'], reward['def']+reward['att'], done, info
        '''

        if action_def is None:
            action_def = np.random.randint(0, 9, dtype=np.int64)
            # print("defender random!!!")
            # action_def = 4 # 5
        if action_att is None:
            action_att = np.random.randint(0, 9, dtype=np.int64)
            # print("attacker random!!!")
            # action_att = 4 # 5
        # print("action_def", action_def, "action_att", action_att)

        # pybullet environment
        self.roundBegin(action_def, action_att)
        self.moveDrones()

        # hyperGame environment state
        # def
        mission_complete_ratio = self.system.scanCompletePercent()
        state_def = [self.system.mission_duration, mission_complete_ratio]
        # att
        # attack success ratio
        if self.attacker.att_counter == 0:
            att_succ_rate = 0
        else:
            att_succ_rate = self.attacker.att_succ_counter/self.attacker.att_counter
        # set of attacker's received signal
        att_signal_set = []
        for id in range(self.system.num_MD+self.system.num_HD):
            if id in self.attacker.obs_sig_dict:
                att_signal_set.append(self.attacker.obs_sig_dict[id])
            else:
                att_signal_set.append(self.attacker.undetect_dbm)

        # state_att = [self.system.mission_duration, att_succ_rate] + att_signal_set
        state_att = [self.system.mission_duration, self.attacker.att_succ_counter, self.attacker.att_counter] + att_signal_set
        # combine
        state = {}
        state['def'] = np.array(state_def)
        state['att'] = np.array(state_att)


        # Defender's Reward
        w_MS = 1.0/3.0
        w_AC = 1.0/3.0
        w_MT = 1.0/3.0
        # if mission_complete_ratio:
        #     reward_def = w_MS * mission_complete_ratio + w_AC * self.system.aliveDroneCount() / (
        #                 self.system.num_MD + self.system.num_HD) + w_MT * self.system.mission_duration / self.system.mission_duration_max
        # else:
        #     reward_def = 0
        energy_HD = self.system.HD_one_round_consume()
        energy_MD = self.system.MD_one_round_consume()
        # mission_effect = w_MS * mission_complete_ratio + w_AC * self.system.aliveDroneCount() / (
        #         self.system.num_MD + self.system.num_HD) + w_MT * (1 - (self.system.mission_duration / self.system.mission_duration_max)) - energy_HD
        # reward_def = math.exp(-1/mission_effect)    # old reward function
        N_AC = len(self.system.MD_connected_set) + len(self.system.HD_connected_set)
        reward_def = math.exp(-1/N_AC) if N_AC else 0


        # Attacker's Reward
        # w_AS = 0.9  # weight for attack success
        # w_AC = 0.05  # weight for attack cost
        # w_SC = 0.05  # weight for number of cell been scanned
        one_round_att_counter = float(self.attacker.att_counter)
        one_round_att_succ_counter = float(self.attacker.att_succ_counter)
        cell_complete_count = self.system.scanCompletePercent()
        # one_round_att_counter = 0   # This is a Test, TODO: remove it (I think we can remove this as the attack success rate is fixed)
        # reward_att = w_AS * one_round_att_succ_counter - w_AC * one_round_att_counter - w_SC * cell_complete_count
        # reward_att = one_round_att_counter
        # mission_effect_att = w_MS * mission_complete_ratio + w_AC * self.system.aliveDroneCount() / (
        #         self.system.num_MD + self.system.num_HD) + w_MT * (
        #                              1 - self.system.mission_duration / self.system.mission_duration_max)
        # reward_att = 1 - math.exp(-1/mission_effect_att)
        reward_numerate = float(1+(self.system.num_MD + self.system.num_HD-self.system.aliveDroneCount()))
        reward_demonirate = float(self.system.mission_duration)
        reward_att = 1 - math.exp(-(reward_numerate/reward_demonirate))


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
        reward['att'] = reward_att

        if self.system.is_mission_Not_end():
            done = False
        else:
            done = True
            self.close_env()  # close client for avoiding client limit error
            print("--- game duration: %s seconds ---" % round(time.time() - self.start_time, 1))


        info = {}
        # info['att_succ_rate'] = att_succ_rate
        info['action_def'] = action_def
        info['action_att'] = action_att
        info['att_succ_counter'] = self.attacker.att_succ_counter
        info['att_counter'] = self.attacker.att_counter
        info["mission_condition"] = self.system.mission_condition
        total_energy_consump = 0 # energy consumption of all drones
        for HD in self.system.HD_set:
            total_energy_consump += HD.accumulated_consumption
        for MD in self.system.MD_set:
            total_energy_consump += MD.accumulated_consumption
        info["total_energy_consump"] = total_energy_consump
        info["scan_percent"] = self.system.scanCompletePercent()
        info["att_reward_0"] = reward_numerate
        info["att_reward_1"] = - reward_demonirate
        info["att_reward_2"] = - reward_numerate/reward_demonirate
        info["def_reward_0"] = w_MS * mission_complete_ratio
        info["def_reward_1"] = w_AC * self.system.aliveDroneCount() / (self.system.num_MD + self.system.num_HD)
        info["def_reward_2"] = w_MT * self.system.mission_duration / self.system.mission_duration_max
        info["MDHD_active_num"] = len(self.system.MD_set) + len(self.system.HD_set)
        info["MDHD_connected_num"] = len(self.system.MD_connected_set) + len(self.system.HD_connected_set)
        info["mission_complete_rate"] = mission_complete_ratio
        info["remaining_time"] = 1 - (self.system.mission_duration / self.system.mission_duration_max)
        info["energy_HD"] = energy_HD
        info["energy_MD"] = energy_MD
        info["recorded_max_RLD_down_time"] = self.system.recorded_max_RLD_down_time
        info["alive_MD_num"] = self.system.aliveMDcount()
        info["alive_HD_num"] = self.system.aliveHDcount()

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

        # for MD update next destination
        if self.frameN <= self.update_freq:  # first round doesn't check scan_map
            self.defender.update_MD_next_destination_no_scanMap()
        else:
            self.defender.update_MD_next_destination()

        # avoid HD crash when creating
        if self.frameN <= self.update_freq:
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
        for _ in range(self.update_freq):
            self.updateTempLoca(self.system.MD_set, self.ARGS.control_freq_hz)
            self.updateTempLoca(self.system.HD_set, self.ARGS.control_freq_hz)
            self.updateTempLoca(self.system.RLD_set, self.ARGS.control_freq_hz)

            obs, reward, done, info = self.env.step(self.pybullet_action)

            # scan count, and check if crashed
            self.system.MD_environment_interaction(obs)
            self.system.HD_environment_interaction(obs)

            # energy consumption of MD and HD
            self.system.battery_consume()

            # execute pybullet_action
            for MD in self.system.MD_set:
                if MD.crashed:
                    continue
                self.pybullet_action[str(MD.ID)], _, _ = self.ctrl[MD.ID].computeControlFromState(
                    control_timestep=self.CTRL_EVERY_N_STEPS * self.env.TIMESTEP,
                    state=obs[str(MD.ID)]["state"],
                    target_pos=MD.xyz_temp)
            for HD in self.system.HD_set:
                self.pybullet_action[str(HD.ID)], _, _ = self.ctrl[HD.ID].computeControlFromState(
                    control_timestep=self.CTRL_EVERY_N_STEPS * self.env.TIMESTEP,
                    state=obs[str(HD.ID)]["state"],
                    target_pos=HD.xyz_temp)

            self.frameN += 1

            #### Sync the simulation ###################################
            if self.ARGS.gui:
                sync(self.frameN, self.START, self.env.TIMESTEP)

        # update Drone's neighbor table
        self.system.update_neighbor_table()
        # update Drone's fleet table
        self.system.update_RLD_connection()
        return

    def render(self, *args):
        pass


    def updateTempLoca(self, drone_set, control_freq_hz: int):
        for drone in drone_set:
            if drone.crashed:
                continue
            vector_per_frame = (drone.xyz - drone.xyz_temp) / (
                    2 * control_freq_hz)  # multiple 2 to slow down drone speed for a long distance travel

            # avoid too fast movement. It cause crash
            vector_magnitude = np.linalg.norm(vector_per_frame)
            if vector_magnitude > drone.speed_per_frame_max:
                vector_per_frame = vector_per_frame / (vector_magnitude / drone.speed_per_frame_max)
            drone.xyz_temp = vector_per_frame + drone.xyz_temp

    def drones_distance(self, drone_x_location, drone_y_location):
        distance_squre = np.square(drone_x_location - drone_y_location)
        return np.sqrt(np.sum(distance_squre))
