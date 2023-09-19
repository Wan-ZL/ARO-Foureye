'''
Project     ：Drone-DRL-HT 
File        ：HT_defender.py
Author      ：Zelin Wan
Date        ：3/27/23
Description : Hypergame Theory Agent for Defender
'''
import math
import numpy as np

from HT_agent import HypergameTheoryAgent


class HypergameTheoryDefender(HypergameTheoryAgent):
    def __init__(self, env, **kwargs):
        HypergameTheoryAgent.__init__(self, action_space=env.action_space['def'].n,
                                      oppo_action_space=env.action_space['att'].n, **kwargs)
        self.env = env
        self.accumu_att_honeyDrone_counter = 0  # accumulated attack on honey drone counter
        self.target_set_size = np.zeros(self.oppo_action_space) # the size of the target set of the attacker
        self.connected_drone_num_accumulate = np.ones(self.action_space) * (self.env.system.num_MD + self.env.system.num_HD)  # accumulated connected drone number per strategy
        self.connected_drone_counter = np.ones(self.action_space)  # connected drone number per strategy

    def Obser_subgame(self):
        '''
        Defender observes the subgame. Defender always know true subgame, because it controls the signal strength.
        :return:
        '''
        def_signal_strength = self.env.defender.strategy2signal_set[self.self_strategy]
        self.subgame = self.dBm2subgameIndex(def_signal_strength)

    def update_connected_drone_num(self):
        self.connected_drone_num_accumulate[self.self_strategy] += len(self.env.system.MD_connected_set) + len(self.env.system.HD_connected_set)
        self.connected_drone_counter[self.self_strategy] += 1


    def update_impact(self):
        '''
        Update the impact of the defense strategy according to Eq. \ref{eq:defense-impact} in the paper
        :return:
        '''

        self.update_connected_drone_num()
        connectivity = (self.connected_drone_num_accumulate[self.self_strategy] / self.connected_drone_counter[self.self_strategy]) / (self.env.system.num_MD + self.env.system.num_HD)

        vul_sum = self.target_set_size[self.oppo_strategy] * self.env.attacker.attack_success_prob
        self.impact[self.self_strategy, self.oppo_strategy] = 1 - (vul_sum / self.env.attacker.max_att_budget) + connectivity

    def update_cost(self):
        '''
        Update the cost of the defense strategy according to Eq. \ref{eq:defense-cost} in the paper
        :return:
        '''
        self.cost[self.self_strategy, self.oppo_strategy] = np.exp((self.self_strategy + 1) - 10)
        # self.cost[self.self_strategy, self.oppo_strategy] = np.exp((self.self_strategy + 1) - 10) + \
        #                                                     self.oppo_impact[self.self_strategy, self.oppo_strategy]
        # Below is the old code:
        # self.cost[self.self_strategy, self.oppo_strategy] = \
        #     ((self.self_strategy + 1) / 10) + self.oppo_impact[self.self_strategy, self.oppo_strategy]

    def update_oppo_cost(self):
        '''
        Update the cost of the defense strategy according to Eq. \ref{eq:defender-perceived-attack-cost} in the paper
        :return:
        '''
        self.oppo_cost[self.self_strategy, self.oppo_strategy] = \
            self.target_set_size[self.oppo_strategy] / self.env.attacker.max_att_budget

    def update_uncertainty(self):
        '''
        Update the uncertainty of the defender according to Eq. \ref{eq:defender-uncertainty} in the paper
        :return:
        '''
        preset_lambda = 0.3 #0.1  # pre-defined parameter
        # self.accumu_att_honeyDrone_counter += self.env.attacker.att_honeyDrone_counter
        self.uncertainty = math.exp(-preset_lambda * self.env.attacker.att_honeyDrone_counter)
