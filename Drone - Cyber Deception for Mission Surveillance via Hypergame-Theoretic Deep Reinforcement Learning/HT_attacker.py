'''
Project     ：Drone-DRL-HT 
File        ：HT_attacker.py
Author      ：Zelin Wan
Date        ：3/27/23
Description : Hypergame Theory Agent for Attacker
'''
import math
import random
import numpy as np

from HT_agent import HypergameTheoryAgent


class HypergameTheoryAttacker(HypergameTheoryAgent):
    def __init__(self, env, **kwargs):
        HypergameTheoryAgent.__init__(self, action_space=env.action_space['att'].n,
                                      oppo_action_space=env.action_space['def'].n, **kwargs)
        self.env = env
        # this is attacker's deception detection ability randomly selected in range [0, 0.5]
        self.attacker_detectability = random.uniform(0, 0.5)
        self.accumu_att_succ_counter = 0  # accumulated attack success counter.
        self.target_set_size = np.zeros(self.oppo_action_space)  # the size of the target set of the defender
        self.att_succ_ratio = 1

    # def getAttributeNeedSave(self):
    #     return {'attacker_detectability': self.attacker_detectability,
    #             'accumu_att_succ_counter': self.accumu_att_succ_counter}

    def Obser_subgame(self):
        '''
        Attacker observes the subgame. Attacker observes the subgame according to the uncertainty.
        :return:
        '''
        if self.uncertainty < random.random() and self.env.attacker.obs_sig_dict:
            # get averaged observed signal strength
            obser_sig_list = []
            for drone_id, signal_strength in self.env.attacker.obs_sig_dict.items():
                obser_sig_list.append(signal_strength)
            ave_obser_sig = sum(obser_sig_list) / len(obser_sig_list)
            # convert the signal strength to subgame index
            self.subgame = self.dBm2subgameIndex(ave_obser_sig)
        else:
            self.subgame = 0

    def update_impact(self):
        '''
        Update the impact of the attack strategy according to Eq. \ref{eq:attack-impact} in the paper
        :return:
        '''
        # Get attack success ratio (ASR' in the paper)
        if self.env.attacker.att_counter_accumulate == 0:
            self.att_succ_ratio = 1
        else:
            self.att_succ_ratio = self.env.attacker.att_succ_counter_accumulate / self.env.attacker.att_counter_accumulate


        weight = np.arange(1, 11, 1) * 0.3
        self.impact[:, self.oppo_strategy] = weight * self.target_set_size * self.att_succ_ratio / self.env.attacker.max_att_budget

        # use drone signal to determine criticality
        # for strat_id in range(self.action_space):
        #     denom = 0
        #     for drone in self.env.attacker.S_target_dict[strat_id]:
        #         obs_sig = self.env.attacker.obs_sig_dict[drone.ID]
        #
        #         criticality = self.signal2Criticality(obs_sig)
        #         denom += criticality * self.att_succ_ratio
        #
        #     self.impact[strat_id, self.oppo_strategy] = denom / self.env.attacker.max_att_budget

        # Below is the old code:
        # weight = 1
        # self.impact[:, self.oppo_strategy] = weight * self.target_set_size * self.att_succ_ratio / self.env.attacker.max_att_budget

        # Below is the old code:
        # crash_counter = 0
        # for drone in self.env.attacker.target_set:
        #     if drone.crashed:
        #         crash_counter += 1
        # self.impact[self.self_strategy, self.oppo_strategy] = crash_counter / self.env.attacker.max_att_budget

    def update_cost(self):
        '''
        Update the cost of the attack strategy according to Eq. \ref{eq:attack-cost} in the paper
        :return:
        '''
        self.cost[:, self.oppo_strategy] = np.exp(self.target_set_size - self.env.attacker.max_att_budget)

        # Below is the old code:
        # self.cost[self.self_strategy, self.oppo_strategy] = self.target_set_size[self.self_strategy] / self.env.attacker.max_att_budget

    def update_oppo_cost(self):
        '''
        Update the cost of the attack strategy according to Eq. \ref{eq:attacker-perceived-defense-cost} in the paper
        :return:
        '''
        self.oppo_cost[self.self_strategy, self.oppo_strategy] = ((self.oppo_strategy + 1) / 10) + self.impact[
            self.self_strategy, self.oppo_strategy]

    def update_uncertainty(self):
        '''
        Update the uncertainty of the attacker according to Eq. \ref{eq:attacker-uncertainty} in the paper
        :return:
        '''
        preset_lambda = 1  # pre-defined parameter
        # self.accumu_att_succ_counter += self.env.attacker.att_succ_counter_one_round
        self.uncertainty = math.exp(
            -preset_lambda * self.attacker_detectability * self.env.attacker.att_succ_counter_accumulate)
        # print("accumu_att_succ_counter: ", self.accumu_att_succ_counter)
        # print("uncertainty: ", self.uncertainty)
