from model_player import player_model
import random
import numpy as np
import torch
from collections import defaultdict

class attacker_model(player_model):
    def __init__(self, system, max_att_budget=5, defense_strategy=0):
        player_model.__init__(self, system)
        # randomly set to target area when create
        self.xyz = np.array([int(system.map_size/2) ,int(system.map_size/2), 0]) #np.array([random.randrange(1,system.map_cell_number+1), random.randrange(1,system.map_cell_number+1), 0])
        self.obs_sig_dict = {}      # key: drone ID, value: observed signal strength
        self.S_target_dict = defaultdict(list)
        self.observe()                  # observe environment and add value to 'obs_sig_dict' and 'S_target_dict'
        self.compromise_record = {}     # key: att_stra (observed signal), value: def_stra (since attacker doesn't know actual signal, we use observed signal)
        self.num_att_stra = 9                    # number of attacker strategy
        self.num_def_stra = 9                    # number o f defender strategy
        self.success_record = np.zeros((system.num_MD + system.num_HD, self.num_def_stra))  # row drone ID, column: def_stra
        self.failure_record = np.zeros((system.num_MD + system.num_HD, self.num_def_stra))
        self.strategy = 10                       # index range (0,9) maps to attack strategy range (1,10)
        self.number_of_strategy = 10            # total number of strategy
        self.strategy2signal_set = [(-100,-98.1), (-98.1,-96.1), (-96.1,-93.8), (-93.8,-91.1), (-91.1,-87.9), (-87.9,-84.0), (-84.0,-79.0), (-79.0,-72.0), (-72.0,-60), (-60,20)]
        self.undetect_dbm = -101
        # condition edit in 'def observe()'. It convert signal strength to strategy index
        self.target_set = []
        self.epsilon = 0.5                      # variable used in determine target range
        self.attack_success_prob = 0.43          # attack success rate of each attack on each drone  # use CVSS from android device https://www.cvedetails.com/cve/CVE-2021-39804/
        self.defense_strategy = defense_strategy  # the strategy used by defender to defend against attacker (0: HD+dynamic signal, 1: IDS+static signal)
        if self.defense_strategy == 1:
            self.attack_success_prob = self.attack_success_prob * 0.9  # IDS can reduce attack success rate by 10%
        self.attack_RLD_prob = 0.0001           # attack success rate of RLD
        self.att_counter = 0    # number of attack launched in a round
        self.att_succ_counter = 0  # count the number of drone compromised in a round
        self.max_att_budget = max_att_budget     # the maximum number of attack can launch in a round


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
        self.obs_sig_dict = {}       # key: drone ID, value: observed signal strength
        self.S_target_dict = defaultdict(list)    # key: observed signal level, value: drone classes
        # conditions = lambda x: {
        #     x < -100: -1, -100 <= x < -98.1: 0, -98.1 <= x < -96.1: 1, -96.1 <= x < -93.8: 2, -93.8 <= x < -91.1: 3,
        #     -91.1 <= x < -87.9: 4,
        #     -87.9 <= x < -84.0: 5, -84.0 <= x < -79.0: 6, -79.0 <= x < -72.0: 7, -72.0 <= x < -60: 8, -60 <= x <= 20: 9,
        #     20 < x: -1
        # }

        # observe signal strength of all drones in the system
        distance_dict = {}
        # observe MD
        for MD in self.system.MD_mission_set:   # only consider MD in mission and not crashed
            distance = self.system.calc_distance(self.xyz, MD.xyz_temp)
            distance_dict[MD.ID] = distance
            obs_signal = self.system.observed_signal(MD.signal, distance)
            self.obs_sig_dict[MD.ID] = obs_signal
            strategy_index = self.signal2strategy(obs_signal)
            self.S_target_dict[strategy_index] = self.S_target_dict[strategy_index] + [MD]
        # observe HD
        for HD in self.system.HD_mission_set:         # we consider crashed drone here
            distance = self.system.calc_distance(self.xyz, HD.xyz_temp)
            distance_dict[HD.ID] = distance
            obs_signal = self.system.observed_signal(HD.signal, distance)
            self.obs_sig_dict[HD.ID] = obs_signal
            strategy_index = self.signal2strategy(obs_signal)
            self.S_target_dict[strategy_index] = self.S_target_dict[strategy_index] + [HD]
        # observe RLD
        distance = self.system.calc_distance(self.xyz, self.system.RLD.xyz_temp)
        distance_dict[self.system.RLD.ID] = distance
        obs_signal = self.system.observed_signal(self.system.RLD.signal, distance)
        self.obs_sig_dict[self.system.RLD.ID] = obs_signal
        strategy_index = self.signal2strategy(obs_signal)
        self.S_target_dict[strategy_index] = self.S_target_dict[strategy_index] + [self.system.RLD]


        if self.print: print("attacker observed:", self.obs_sig_dict)
        if self.print: print("attacker obs distance:", distance_dict)      # TODO: check if distance-signal function are correct

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
                        ai[att_stra, def_stra] += (self.success_record[drone.ID, def_stra]/ (self.success_record[drone.ID, def_stra] + self.failure_record[drone.ID, def_stra]))

        if self.print: print("ai", ai)
        if self.print: print("max_set", max_set)

        ai = ai/max_set
        return ai


    def action(self):
        # return: number of drone compromised in one action
        if self.print: print("attacker strategy:", self.strategy, "signal", self.strategy2signal_set[self.strategy])
        target_set = self.S_target_dict[self.strategy]

        if len(target_set) > self.max_att_budget:
            # if exceed budget limit, randomly select some of them
            target_set = random.sample(target_set, self.max_att_budget)

        self.att_counter = 0        # reset counter before action execute
        self.att_succ_counter = 0   # reset counter before action execute
        for drone in target_set:
            if self.print: print("attacking", drone.ID, drone.type)
            self.att_counter += 1
            # attack MD
            if drone.type == "MD":
                if self.defense_strategy == 2:
                    # if MD use strategy 2, it will not be compromised. Instead, the drone cannot continue mission (memory full)
                    drone.memory_full = True
                else:
                    # if self.defense_strategy == 1 or self.defense_strategy == 3:
                    #     # use no fixed random seed for avoid the error of no success attacks
                    #     temp_dice = torch.rand((1,)).item()
                    # else:
                    #     temp_dice = random.uniform(0, 1)
                    temp_dice = torch.rand((1,)).item()

                    # for other strategies, MD will be compromised
                    if temp_dice < self.attack_success_prob:
                        if self.print: print("attack success:", drone)
                        drone.xyz[2] = 0
                        drone.xyz_temp[2] = 0
                        drone.crashed = True
                        self.att_succ_counter += 1
            # attack RLD
            elif drone.type == "RLD":
                if random.uniform(0, 1) < self.attack_RLD_prob:
                    if self.print: print("attack success:", drone)
                    self.system.crashed_RLD_counter += 1
                    self.system.RLD_down_time = 5
                    self.att_succ_counter += 1
            # attack HD
            else:
                # HD won't be compromised
                pass



















