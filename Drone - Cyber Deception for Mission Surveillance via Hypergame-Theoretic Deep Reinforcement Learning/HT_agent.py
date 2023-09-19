'''
Project     ：Drone-DRL-HT 
File        ：HT_agent.py
Author      ：Zelin Wan
Date        ：3/27/23
Description : Frameward for agent that based on Hypergame Theory
'''
import os
import random
import time
import numpy as np

from torch.utils.tensorboard import SummaryWriter

class HypergameTheoryAgent():
    def __init__(self, action_space, oppo_action_space, subgame_space=4, writer=None):
        '''
        Initialize the agent
        :param action_space: player's action space
        :param oppo_action_space: opponent's action space
        :param role: 'def' for defender, 'att' for attacker
        :param subgame_space: the number of subgames
        :param writer: tensorboard writer
        '''
        self.env = None # initialize environment in def/att agent
        self.action_space = action_space
        self.oppo_action_space = oppo_action_space
        self.subgame_space = subgame_space
        self.utility = np.ones((action_space, oppo_action_space)) / (action_space)  # make the sum of strategy for
        # each opponent action to be 1
        # TODO: consider impact initial to 1 if performance not good.
        average_initial_utility = False # True for average initial utility, False for max initial utility. Under average
        # initial utility, if a strategy receive a high utility, it will be more likely to be selected. So initial
        # utility with maximum utility gives the strategy that's no been selected a chance to be selected.
        if average_initial_utility:
            self.impact = np.ones((action_space, oppo_action_space)) / (action_space * oppo_action_space)
            self.oppo_impact = np.ones((action_space, oppo_action_space)) / (action_space * oppo_action_space)
            self.cost = np.ones((action_space, oppo_action_space)) / (action_space * oppo_action_space)
            self.oppo_cost = np.ones((action_space, oppo_action_space)) / (action_space * oppo_action_space)
        else:
            self.impact = np.ones((action_space, oppo_action_space))
            self.oppo_impact = np.zeros((action_space, oppo_action_space))
            self.cost = np.zeros((action_space, oppo_action_space))
            self.oppo_cost = np.ones((action_space, oppo_action_space))

        self.CMS = np.ones((subgame_space, oppo_action_space)) / (subgame_space * oppo_action_space)
        self.Observation_record_counter = np.ones((subgame_space, oppo_action_space))  # this is $gamma$ in the paper.
        # Start from 1 to avoid divide by 0
        self.subgame_prob = np.ones(subgame_space) / subgame_space
        self.stra_belief = np.ones(oppo_action_space) / oppo_action_space  # this is 'S' in the paper
        self.uncertainty = 1
        self.expected_utility = np.ones(action_space) / action_space
        self.worst_expected_utility = np.ones(action_space) / action_space
        self.hypergame_expected_utility = np.ones(action_space) / action_space
        self.HEU_prob = np.ones(action_space) / action_space
        self.HEU_prob_record = np.empty((0, action_space))  # record the HEU prob for each step
        self.HEU_prob_record = np.append(self.HEU_prob_record, self.HEU_prob[np.newaxis], axis=0) # initialize the record

        self.self_strategy = random.randint(0, action_space - 1)  # range from 0 to 9. Initialize with random
        self.oppo_strategy = random.randint(0, self.oppo_action_space - 1)  # range from 0 to 9. Initialize with random
        self.subgame = 0  # range from 0 to 3. 0 means full game, and initialize the subgame to be 0
        self.writer = writer
        self.writer_counter = 0
        self.target_set_size = None # the size of target set of attacker (initialize this in def/att agent)
        self.action_counter = np.zeros((action_space, oppo_action_space)) # the number of times that each strategy is selected
        self.action_counter[self.self_strategy, self.oppo_strategy] += 1

    def get_u_ai_ac_di_dc(self):
        return [self.utility[self.self_strategy, self.oppo_strategy], self.impact[self.self_strategy, self.oppo_strategy],
                self.cost[self.self_strategy, self.oppo_strategy], self.oppo_impact[self.self_strategy, self.oppo_strategy],
                self.oppo_cost[self.self_strategy, self.oppo_strategy]]

    def update_target_set_size(self):
        if self.target_set_size is None:
            raise ValueError("The target set size is not initialized, please check function 'update_target_set_size'")
        if self.env is None:
            raise ValueError("The environment is not initialized, please check function 'update_target_set_size'")

        for i in range(len(self.target_set_size)):
            self.target_set_size[i] = len(self.env.attacker.S_target_dict[i])

    def signal2Criticality(self, signal):
        '''
        Normalize the signal in range [-100, 20] to [0, 1]
        :param signal: signal to be normalized
        :return:
        '''

        return (signal + 100) / 120

    def write2Tensorboard(self):
        '''
        Write the data to tensorboard
        :return:
        '''
        if self.writer is not None:
            self.writer.add_scalar("Uncertainty", self.uncertainty, self.writer_counter)
            self.writer.add_scalar("Subgame", self.subgame, self.writer_counter)
            self.writer.add_scalar("Oppo_Strategy", self.oppo_strategy, self.writer_counter)

            for i in range(self.action_space):
                self.writer.add_scalar("Impact_stra_" + str(i), self.impact[i, self.oppo_strategy], self.writer_counter)
                self.writer.add_scalar("Oppo_Impact_stra_" + str(i), self.oppo_impact[i, self.oppo_strategy], self.writer_counter)
                self.writer.add_scalar("Cost_stra_" + str(i), self.cost[i, self.oppo_strategy], self.writer_counter)
                self.writer.add_scalar("Oppo_Cost_stra_" + str(i), self.oppo_cost[i, self.oppo_strategy], self.writer_counter)
                self.writer.add_scalar("Utility_stra_" + str(i), self.utility[i, self.oppo_strategy], self.writer_counter)
                self.writer.add_scalar("Expected_Utility_stra_" + str(i), self.expected_utility[i], self.writer_counter)
                self.writer.add_scalar("Worst_Expected_Utility_stra_" + str(i), self.worst_expected_utility[i], self.writer_counter)
                self.writer.add_scalar("Hypergame_Expected_Utility_stra_" + str(i), self.hypergame_expected_utility[i], self.writer_counter)
                self.writer.add_scalar("HEU_Prob_stra_" + str(i), self.HEU_prob[i], self.writer_counter)
                self.writer.add_scalar("Target_Set_Size_stra_" + str(i), self.target_set_size[i], self.writer_counter)

            for drone_id in range(20):
                if drone_id in self.env.attacker.distance_dict.keys():
                    self.writer.add_scalar("Distance_drone_" + str(drone_id), self.env.attacker.distance_dict[drone_id], self.writer_counter)
                else:
                    self.writer.add_scalar("Distance_drone_" + str(drone_id), -100, self.writer_counter)

            self.writer_counter += 1


    def getAttributeNeedSave(self):
        '''
        Get the attributes that need to be saved
        :return:
        '''
        pass

    def dBm2subgameIndex(self, dBm):
        '''
        Convert the received signal strength to subgame index
        :param dBm:
        :return:
        '''
        if self.subgame_space != 4:
            # The subgame space is not 4, raise error
            raise ValueError("The subgame space is not 4, please check the function 'dBm2subgameIndex' in HT_agent.py")

        if -100.0 < dBm <= -93.8:
            return 1
        elif -93.8 < dBm <= -79.0:
            return 2
        elif -79.0 < dBm <= 20.0:
            return 3
        else:
            return 0

    def act(self):
        '''
        Choose the strategy according to the probability distribution of the hypergame expected utility
        :return: selected strategy
        '''
        self.refresh_HEU_for_each_strategy()

        # convert the hypergame expected utility to probability distribution with min-max normalization
        if np.max(self.hypergame_expected_utility) != np.min(self.hypergame_expected_utility):
            self.HEU_prob = (self.hypergame_expected_utility - np.min(self.hypergame_expected_utility)) / (
                    np.max(self.hypergame_expected_utility) - np.min(self.hypergame_expected_utility))
        self.HEU_prob += 0.01  # avoid the probability distribution to be 0
        self.HEU_prob /= np.sum(self.HEU_prob)  # normalize the probability distribution
        # print("HEU: ", self.hypergame_expected_utility)
        # print("HEU_prob: ", self.HEU_prob)
        # print("Observation_record_counter: ", self.Observation_record_counter)

        # choose the strategy according to the probability distribution
        self.self_strategy = np.random.choice(self.action_space, p=self.HEU_prob)

        # record the data
        self.HEU_prob_record = np.append(self.HEU_prob_record, self.HEU_prob[np.newaxis], axis=0)
        self.action_counter[self.self_strategy, self.oppo_strategy] += 1

        # Write the data to tensorboard
        self.write2Tensorboard()

        return self.self_strategy

    def observe(self, oppo_strategy):
        '''
        Observe the opponent's strategy and the subgame when round ends
        :param oppo_strategy:
        :return:
        '''
        self.Obser_oppo_strategy(oppo_strategy)
        self.Obser_subgame()

    def Obser_oppo_strategy(self, oppo_strategy):
        '''
        Observe opponent's strategy when round ends
        :param oppo_strategy:
        :return:
        '''
        if self.uncertainty < random.random():
            self.oppo_strategy = oppo_strategy
        else:
            self.oppo_strategy = random.randint(0, self.oppo_action_space - 1)

    def Obser_subgame(self):
        '''
        Observe the subgame when round ends
        :return:
        '''
        pass

    def update_utility(self):
        self.utility = (self.impact + self.oppo_cost) - (self.cost + self.oppo_impact)

    def update_impact(self):
        pass

    def update_cost(self):
        pass

    def update_oppo_impact(self):
        self.oppo_impact = 1 - self.impact

    def update_oppo_cost(self):
        pass

    def update_obser_record_counter(self):
        '''
        This function is used to update the observation on the opponent's action
        :param subgame:
        :param oppo_action:
        :return:
        '''
        self.Observation_record_counter[self.subgame, self.oppo_strategy] += 1

    def update_CMS(self):
        self.CMS = self.Observation_record_counter / self.Observation_record_counter.sum(axis=1, keepdims=True)

    def update_subgame_prob(self):
        self.subgame_prob = self.Observation_record_counter.sum(axis=1) / self.Observation_record_counter.sum()

    def update_stra_belief(self):
        '''
        Update player's belief on opponent's strategy according to Eq. \ref{Eq:attacker-belief} and
        Eq. \ref{Eq:defender-belief}.
        :return:
        '''
        self.stra_belief = self.subgame_prob @ self.CMS

    def update_uncertainty(self):
        pass

    def update_expected_utility(self):
        self.expected_utility = self.utility @ self.stra_belief # equals to self.stra_belief@self.utility.transpose()

    def update_worst_expected_utility(self):
        self.worst_expected_utility = self.oppo_action_space * (self.utility * self.stra_belief).min(axis=1)

    def update_hypergame_expected_utility(self):
        self.hypergame_expected_utility = (1 - self.uncertainty) * self.expected_utility \
                                          + self.uncertainty * self.worst_expected_utility

    def refresh_HEU_for_each_strategy(self):
        '''
        Calculate the hypergame expected utility for each strategy according to section \ref{subsec:hypergame-attack} and \ref{subsec:hypergame-defense}
        in the paper.
        :return:
        '''
        self.update_target_set_size()
        self.update_impact()
        self.update_oppo_impact()
        self.update_cost()
        self.update_oppo_cost()
        self.update_utility()
        self.update_obser_record_counter()
        self.update_CMS()
        self.update_subgame_prob()
        self.update_stra_belief()
        self.update_uncertainty()
        self.update_expected_utility()
        self.update_worst_expected_utility()
        self.update_hypergame_expected_utility()
