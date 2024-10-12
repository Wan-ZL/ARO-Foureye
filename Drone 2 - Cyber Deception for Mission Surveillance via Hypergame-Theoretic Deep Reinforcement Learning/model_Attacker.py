'''
Project     ：Drone-DRL-HT 
File        ：model_Attacker.py
Author      ：Zelin Wan
Date        ：2/7/23
Description : 
'''

from model_player import player_model
import random
import numpy as np
from collections import defaultdict


class attacker_model(player_model):
    def __init__(self, system, max_att_budget=5, def_select_method='fixed'):
        player_model.__init__(self, system)
        # randomly set to target area when create
        self.xyz = np.array([int(system.map_size * 2 / 3), int(system.map_size * 2/ 3), 0])# attacker locate in the top right corner
        # self.xyz = np.array([int(system.map_size / 2), int(system.map_size / 2),
        #                      0])  # np.array([random.randrange(1,system.map_cell_number+1), random.randrange(1,system.map_cell_number+1), 0])
        self.obs_sig_dict = {}  # key: drone ID, value: observed signal strength
        self.S_target_dict = defaultdict(list)  # Key: strategy, value: list of drone ID. Observed drones. Classified based on observed signal strength
        self.observe()  # observe environment and add value to 'obs_sig_dict' and 'S_target_dict'
        self.compromise_record = {}  # key: att_stra (observed signal), value: def_stra (since attacker doesn't know actual signal, we use observed signal)
        self.distance_dict = {}  # key: drone ID, value: distance between attacker and drone
        self.num_att_stra = 9  # number of attacker strategy
        self.num_def_stra = 9  # number o f defender strategy
        self.success_record = np.zeros(
            (system.num_MD + system.num_HD, self.num_def_stra))  # row drone ID, column: def_stra
        self.failure_record = np.zeros((system.num_MD + system.num_HD, self.num_def_stra))
        self.strategy = 9  # index range (0,9) maps to attack strategy 1 to 10 in the paper
        self.number_of_strategy = 10  # total number of strategy
        self.strategy2signal_set = [(-100, -98.1), (-98.1, -96.1), (-96.1, -93.8), (-93.8, -91.1), (-91.1, -87.9),
                                    (-87.9, -84.0), (-84.0, -79.0), (-79.0, -72.0), (-72.0, -60), (-60, 20)]
        self.undetect_dbm = -101
        # condition edit in 'def observe()'. It converts signal strength to strategy index
        self.target_set = []    # a list of drone in the selected target set (based on selected strategy)
        self.def_select_method = def_select_method  # defense method used by defender
        self.epsilon = 0.5  # variable used in determine target range
        self.attack_success_prob = 0.43  # attack success rate of each attack on each drone  # use CVSS from android device https://www.cvedetails.com/cve/CVE-2021-39804/
        if self.def_select_method == 'IDS':
            self.attack_success_prob = self.attack_success_prob * 0.1 #0.5  # IDS can reduce attack success rate by 10%
        self.att_counter_one_round = 0  # number of attack launched in a round
        self.att_succ_counter_one_round = 0  # count the number of drone compromised in a round
        self.att_counter_accumulate = 0  # accumulate the number of attack launched in total
        self.att_succ_counter_accumulate = 0  # accumulate the number of drone compromised in total
        self.att_honeyDrone_counter = 0  # count the number of attack targeting honey drone
        self.max_att_budget = max_att_budget  # the maximum number of attack can launch in a round


    def signal2strategy(self, obs_signal):
        conditions = lambda x: {
            x <= -100: -1, -100 < x <= -98.1: 0, -98.1 < x <= -96.1: 1, -96.1 < x <= -93.8: 2, -93.8 < x <= -91.1: 3,
            -91.1 < x <= -87.9: 4,
            -87.9 < x <= -84.0: 5, -84.0 < x <= -79.0: 6, -79.0 < x <= -72.0: 7, -72.0 < x <= -60: 8, -60 < x <= 20: 9,
            20 < x: -1
        }
        return conditions(obs_signal)[True]

    # observation action
    def observe(self):
        self.S_target_dict = defaultdict(list)  # key: observed signal level, value: drone classes
        self.obs_sig_dict = {}  # key: drone ID, value: observed signal strength
        self.distance_dict = {}  # key: drone ID, value: distance between attacker and drone

        for MD in self.system.MD_mission_set:  # only consider MD in mission and not crashed
            distance = self.system.calc_distance(self.xyz, MD.xyz_temp)

            self.distance_dict[MD.ID] = distance
            obs_signal = self.system.observed_signal(MD.signal, distance)

            self.obs_sig_dict[MD.ID] = obs_signal
            strategy_index = self.signal2strategy(obs_signal)

            self.S_target_dict[strategy_index] = self.S_target_dict[strategy_index] + [MD]

        for HD in self.system.HD_mission_set:  # we consider crashed drone here
            distance = self.system.calc_distance(self.xyz, HD.xyz_temp)
            self.distance_dict[HD.ID] = distance
            obs_signal = self.system.observed_signal(HD.signal, distance)
            self.obs_sig_dict[HD.ID] = obs_signal
            strategy_index = self.signal2strategy(obs_signal)
            self.S_target_dict[strategy_index] = self.S_target_dict[strategy_index] + [HD]

        if self.print: print("attacker observed:", self.obs_sig_dict)
        if self.print: print("attacker obs distance:",
                             self.distance_dict)  # TODO: check if distance-signal function are correct

    def impact(self):
        ai = np.ones((self.num_att_stra, self.num_def_stra), dtype=float) / (self.num_att_stra * self.num_def_stra)
        max_set = 0
        if self.print: print("S_target_dict", self.S_target_dict)
        for att_stra in range(self.num_att_stra):
            # find denominator
            if len(self.S_target_dict[att_stra]) > max_set:
                max_set = len(self.S_target_dict[att_stra])

            # calculate numerator
            for def_stra in range(self.num_def_stra):
                # numerat_sum = 0
                for drone in self.S_target_dict[att_stra]:
                    if self.success_record[drone.ID, def_stra]:
                        ai[att_stra, def_stra] += (self.success_record[drone.ID, def_stra] / (
                                self.success_record[drone.ID, def_stra] + self.failure_record[drone.ID, def_stra]))

        if self.print: print("ai", ai)
        if self.print: print("max_set", max_set)

        ai = ai / max_set
        return ai

    # def select_strategy(self, new_strategy):
    #     self.strategy = 8

    def action(self):
        # return: number of drone compromised in one action
        if self.print: print("attacker strategy:", self.strategy, "signal", self.strategy2signal_set[self.strategy])
        self.target_set = self.S_target_dict[self.strategy]

        if len(self.target_set) > self.max_att_budget:
            # if exceed budget limit, randomly select some of them
            self.target_set = random.sample(self.target_set, self.max_att_budget)

        self.att_counter_one_round = 0  # reset counter before action execute
        self.att_succ_counter_one_round = 0  # reset counter before action execute
        # print("target_set", self.target_set)
        for drone in self.target_set:
            if self.print: print("attacking", drone.ID, drone.type)
            self.att_counter_one_round += 1
            # attack MD
            if drone.type == "MD":
                # only attack not crashed drone
                if random.uniform(0, 1) < self.attack_success_prob:
                    if self.def_select_method == 'CD':
                        drone.memory_full = 50
                    else:
                        # drone.xyz[2] = 0
                        drone.xyz_temp[2] = 0
                        drone.crashed = True
                        self.att_succ_counter_one_round += 1
            elif drone.type == "HD":
                self.att_honeyDrone_counter += 1

            # attack HD/RLD
            else:
                # HD and RLD won't be compromised
                pass

        self.att_counter_accumulate += self.att_counter_one_round
        self.att_succ_counter_accumulate += self.att_succ_counter_one_round



