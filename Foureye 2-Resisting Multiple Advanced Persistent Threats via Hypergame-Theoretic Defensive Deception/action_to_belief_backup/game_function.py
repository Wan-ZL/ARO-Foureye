#!/usr/bin/env python
# coding: utf-8

# In[25]:
from graph_function import update_vul
from graph_function import *

from attacker_function import *
from defender_function import *
from main import display
from networkx import nx
from itertools import count
import concurrent
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import copy
import time
import pickle


# In[1]:


# In[13]:


class game_class:
    def __init__(self,
                 simulation_id,
                 DD_using,
                 uncertain_scheme_att,
                 uncertain_scheme_def,
                 decision_scheme,
                 scheme_name,
                 web_data_upper_vul,
                 Iot_upper_vul):
        print(f"Sim {simulation_id} creating game")
        self.lifetime = 1
        self.CKC_number = 6
        self.strategy_number = 8     # Note: 8 means the ninth strategy disabled, 9 means all strategie used.
        self.use_bundle = True      # Note: False means defender only use one strategy each game
        self.enable_IRS_recheck = False     # True means enable IRS rechecking
        self.enable_IRS_recover = False     # True means enable IRS recovery
        self.new_attacker_probability = 0.1 # 1  # 0 means only one attacker in game.
        self.DD_using = DD_using
        self.decision_scheme = decision_scheme
        self.scheme_name = scheme_name
        self.node_size_multiplier = 1  # 1 means 100 nodes, 5 means 500 nodes
        self.graph = graph_class(
            web_data_upper_vul, Iot_upper_vul, self.node_size_multiplier)
        self.uncertain_scheme_att = uncertain_scheme_att
        self.uncertain_scheme_def = uncertain_scheme_def
        self.collusion_attack_probability = 1  # TODO: add to calc_APV()
        self.attacker_ID = 0
        self.attacker_list = []
        self.attacker_template = attacker_class(self, self.attacker_ID)
        self.attacker_list.append(
            attacker_class(self, self.attacker_ID))  # for avoid index out of range eror.
        self.attacker_number = 1
        self.defender = defender_class(self)
        self.game_over = False
        self.FN = 10  # False Negative for Beta distribution >0.1
        self.TP = 90  # True Positive
        self.TN = 90
        self.FP = 10
        self.rewire_network = 0.1
        # for Experiment Result ⬇️
        self.def_uncertainty_history = []
        self.att_uncertainty_history = []
        self.att_HEU_history = []
        self.def_HEU_history = []
        self.AHEU_per_Strategy_History = []
        self.DHEU_per_Strategy_History = []
        self.att_strategy_counter = []
        self.def_strategy_counter = []
        self.FPR_history = []
        self.TPR_history = []
        self.att_cost_history = []
        self.def_cost_history = []
        self.criticality_hisotry = np.zeros(100000)  # np.zeros(10000)
        self.evict_reason_history = np.zeros(5)
        self.SysFail = [False] * 3
        self.att_EU_C = np.zeros(self.strategy_number)
        self.att_EU_CMS = np.zeros(self.strategy_number)
        self.def_EU_C = np.zeros(self.strategy_number)
        self.def_EU_CMS = np.zeros(self.strategy_number)
        self.att_impact = np.zeros(self.strategy_number)
        self.def_impact = np.zeros(self.strategy_number)
        self.att_HEU_DD_IPI = np.zeros(self.strategy_number)
        self.def_HEU_DD_IPI = np.zeros(self.strategy_number)
        self.NIDS_eviction = np.zeros(4)  # [# of bad, # of good]
        self.att_number = []
        self.att_CKC = []
        self.compromise_probability = []
        self.number_of_inside_attacker = []
        self.all_result_def_obs_action = []
        self.all_result_def_belief = []

    def defender_round(self):
        # save data for train KNN model
        # before_observation_history = self.defender.obs_oppo_strat_history.copy()

        # only observe opponent action in one game
        # self.defender.oppo_strat_in_last_game = np.zeros(len(self.defender.oppo_strat_in_last_game))

        # observe
        self.defender.observe_opponent(self.attacker_list)




        # Observe CKC
        self.defender.decide_CKC_posi(get_CKC_list(self.attacker_list))

        # create bundles
        self.defender.create_bundle(self.DD_using)

        # select strategy bundle
        self.defender.choose_bundle(
            self.attacker_template.strategy_number, self.attacker_template.strat_cost,
            get_averaged_impact(self.attacker_list, self.attacker_template))

        self.defender.execute_strategy(get_network_list(self.attacker_list), get_detect_prob_list(self.attacker_list),
                                       self.graph,
                                       self.FN / (self.TP + self.FN), self.FP / (self.TN + self.FP), self.NIDS_eviction,
                                       get_P_fake_list(self.attacker_list))

        # save data for train KNN
        self.all_result_def_obs_action.append(self.defender.ML_action_save.copy())
        self.all_result_def_belief.append(self.defender.S_j.copy())

        self.defender.update_attribute(get_average_detect_prob(self.attacker_list))
        self.update_graph()

    def attacker_round(self, simulation_id):
        # random.shuffle(self.attacker_list)      # to avoid oder effect. Attackers execute in random order.
        for attacker in self.attacker_list:
            if display:
                print(f"attacker location{attacker.location}")
            if self.game_over:
                print(f"Sim {simulation_id} GAME OVER")
                return

            attacker.observe_opponent(
                self.defender.chosen_strategy_list)

            attacker.choose_strategy(
                self.defender.strategy_number, self.defender.strat_cost, self.defender.impact_record)


            if display:
                print(f"attacker choose: {attacker.chosen_strategy + 1}")
            attack_result = attacker.execute_strategy(self.graph.network, self.defender.network,
                                                      self.node_size_multiplier, self.compromise_probability)


            attacker.update_attribute(self.defender.dec)
            self.update_graph()
            if attack_result:
                if attacker.chosen_strategy == 0 and attacker.CKC_position != 0:
                    pass  # This avoid inside attacker increase stage when Strategy 1 success
                else:
                    attacker.next_stage()
            else:
                attacker.random_moving()
                if display:
                    print(
                        f"attacker move, new location: {attacker.location}")

        return

    def evict_attacker(self, attacker_class):
        print(f"\033[93m Evict Attacker ID: {attacker_class.attacker_ID} \033[0m")
        self.attacker_list.remove(attacker_class)

    def NIDS_detect(self):
        # Positive rate
        #         false_neg_prob = self.FN / (self.TP + self.FN)
        true_pos_prob = self.TP / (self.TP + self.FN)
        false_pos_prob = self.FP / (self.TN + self.FP)

        #         for index in self.graph.network.nodes:
        all_nodes = list(self.graph.network.nodes(data=False))

        for index in all_nodes:
            # ignore evicted node for saving time
            if is_node_evicted(self.graph.network, index):
                continue

            # ignore stealthy attack
            if self.defender.network.nodes[index]["stealthy_status"]:
                self.defender.network.nodes[index]["stealthy_status"] = False
                continue

            # detect is node compromised
            node_is_compromised = False
            if self.graph.network.has_node(index):
                if self.graph.network.nodes[index]["compromised_status"]:
                    if random.random() <= true_pos_prob:
                        node_is_compromised = True
                        self.defender.network.nodes[index]["compromised_status"] = True
                    else:
                        if display:
                            print("IDS: False Negative to compromised node")
                else:
                    if random.random() <= false_pos_prob:
                        if display:
                            print("False Positive to good node")
                        node_is_compromised = True
                        self.defender.network.nodes[index]["compromised_status"] = True

    # IRS evict the compromised node that is labeled by IDS
    def IDS_IRS_evict(self):
        # IDS Eviction
        Th_risk = 7  # pre-set value
        IRS_inspection_prob = 0.5 # 0.9 # probability to accurately detect false positive.

        all_nodes = self.graph.network.nodes(data=False)
        for index in all_nodes:
            if self.defender.network.nodes[index]["compromised_status"] == True:
                # IRS inspection
                if self.graph.network.nodes[index]["compromised_status"] == False:
                    if random.random() < IRS_inspection_prob and self.enable_IRS_recheck:
                        self.defender.network.nodes[index]["compromised_status"] = False
                        continue

                # No-DD means NIDS doesn't remain attacker in system
                if not self.DD_using:
                    if display:
                        print(f"Evict node {index}, No DD using")
                    evict_a_node_without_update_criticality(index, self.graph.network,
                                                self.defender.network, get_network_list(self.attacker_list))
                    # self.NIDS_eviction[experiment_index_record] += 1
                    continue

                if self.graph.network.has_node(index):
                    if self.graph.network.nodes[index]["importance"] > Th_risk:
                        if display:
                            print(f"Evict node {index}, importance > Th_risk")
                        evict_a_node_without_update_criticality(index, self.graph.network,
                                                    self.defender.network,
                                                    get_network_list(self.attacker_list))
                            # self.NIDS_eviction[experiment_index_record] += 1


        # update criticality
        update_criticality(self.graph.network)
        update_criticality(self.defender.network)
        for G_att in get_network_list(self.attacker_list):
            update_criticality(G_att)

    def IRS_recover(self):
        if not self.enable_IRS_recover:
            return


        # decrease "recover_time", only recover the node with "recover_time" = 1
        all_nodes = self.graph.network.nodes(data=False)
        for index in all_nodes:
            # if it's time to recover
            if self.graph.network.nodes[index]["recover_time"] == 1:
                self.graph.network.nodes[index]["recover_time"] -= 1
                recover_the_node(self.graph.network, self.defender.network,
                                 get_attacker_network_list(self.attacker_list), index)
            elif self.graph.network.nodes[index]["recover_time"] > 1:
                self.graph.network.nodes[index]["recover_time"] -= 1

        # add recover time for False Nodes
        recover_threshold = 0.5  # robust threshold 1-vul
        recover_min_time = 2  # time of recovery
        recover_max_time = 10
        # get all non-compromised evicted node
        false_node_list = false_evicted_node_list(self.graph.network)
        for index in false_node_list:
            if self.graph.network.nodes[index]["recover_time"] == 0:
                if (1 - self.graph.network.nodes[index]["normalized_vulnerability"]) > recover_threshold:
                    if self.graph.network.nodes[index]["compromised_status"]:
                        self.graph.network.nodes[index]["recover_time"] = 1
                    else:
                        self.graph.network.nodes[index]["recover_time"] = random.randint(
                            recover_min_time, recover_max_time)


    def update_graph(self):
        update_criticality(self.graph.network)
        update_criticality(self.defender.network)

        update_vul(self.graph.network)
        update_vul(self.defender.network)

        update_en_vul(self.graph.network, self.graph.ev,
                      self.graph.ev_lambda, self.graph.T_rekey)
        update_en_vul(self.defender.network, self.graph.ev,
                      self.graph.ev_lambda, self.graph.T_rekey)

        for attacker in self.attacker_list:
            update_criticality(attacker.network)
            update_vul(attacker.network)
            update_en_vul(attacker.network, self.graph.ev,
                          self.graph.ev_lambda, self.graph.T_rekey)

    def prepare_for_next_game(self):

        self.lifetime += 1

        # remove attacker from list if it's evicted by IRS or successfully exfiltrate data
        for attacker in self.attacker_list:
            if attacker.exfiltrate_data:
                print(f"Attacker {attacker.attacker_ID} exfiltrate data")
                self.evict_attacker(attacker)
                self.evict_reason_history[4] += 1
                continue

            if attacker.location is not None:
                if self.graph.network.nodes[attacker.location]["type"] == 3:  # if in honeypot
                    print("attacker in honeypot")
                    self.evict_reason_history[0] += 1
                    self.evict_attacker(attacker)
                elif self.graph.network.nodes[attacker.location]["evicted_mark"]:  # if the attacker located node is evicted
                    self.evict_attacker(attacker)
                else:
                    continue

        # Beta distribution
        if self.graph.using_honeynet:
            self.TP += 1
            self.TN += 1
        else:
            if any(elem in self.defender.chosen_strategy_list for elem in [4, 5, 6, 7]):
                self.TP += 1
                self.TN += 1
        #             if self.defender.chosen_strategy == 4 or self.defender.chosen_strategy == 5 or self.defender.chosen_strategy == 6 or self.defender.chosen_strategy == 7:

        # rewire graph
        rewire_network(self.graph.network, get_network_list(self.attacker_list),
                       self.defender.network, self.rewire_network)

        # reconnect non-evicted node to server or databse
        node_reconnect(self.graph.network, get_network_list(self.attacker_list),
                       self.defender.network, self.graph.connect_prob)

        # update defender impact (done)
        self.defender.update_defense_impact(get_overall_attacker_impact_per_game(self.attacker_list, self.attacker_template))

        # clean honeypot after each game
        if self.graph.using_honeynet:
            clean_honeynet(self.graph.network, get_network_list(self.attacker_list),
                           self.defender.network)
            self.graph.using_honeynet = False

        for attacker in self.attacker_list:
            # For Double Check: remove attacker if it's location doesn't exist anymore
            if attacker.location is not None:
                if not self.graph.network.has_node(attacker.location):
                    print(f"attacker location: {attacker.location}")
                    self.evict_reason_history[0] += 1
                    self.evict_attacker(attacker)
                    continue

            # remove honeypot in comrpomised list
            for index in attacker.compromised_nodes:
                if not self.graph.network.has_node(index):
                    attacker.compromised_nodes.remove(index)
            # remove honeypot in collection list
            for index in attacker.collection_list:
                if not self.graph.network.has_node(index):
                    attacker.collection_list.remove(index)

    # Add attacker with probability. If no attacker in list, add one.
    def new_attacker(self, simulation_id, defender):
        if random.random() < self.new_attacker_probability or len(self.attacker_list)==0:
            self.attacker_number += 1
            print(
                f"\033[93m Sim {simulation_id} Creating attacker #{self.attacker_number} \033[0m"
            )
            # add new attacker
            self.attacker_ID += 1
            self.attacker_list.append(
                attacker_class(self, self.attacker_ID))
            # defender.reset_attribute(defender.CKC_number) # reset defender (for test)

    def count_number_of_evicted_attacker(self):
        counter = 0
        for attacker in self.attacker_list:
            if attacker.location is not None:
                if not self.graph.network.has_node(attacker.location):
                    print("ERROR!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    print(f"attacker location: {attacker.location}")
                    # draw_graph(self.graph.network)
                    # show_all_nodes(self.graph.network)
                    # raise Exception("Error: attacker location doesn't exist in network")
                if self.graph.network.has_node(attacker.location):
                    if self.graph.network.nodes[attacker.location]["type"] != 3:
                        if self.graph.network.nodes[attacker.location]["evicted_mark"]:  # if the attacker located node is evicted
                            counter+=1

        return counter



    def experiment_saving(self):

        self.def_uncertainty_history.append(self.defender.uncertainty)
        att_uncertain_one_game = []
        for attacker in self.attacker_list:
            att_uncertain_one_game.append(attacker.uncertainty)
        self.att_uncertainty_history.append(att_uncertain_one_game)

        # Att/Def HEU
        att_HEU_one_game = []
        for attacker in self.attacker_list:
            att_HEU_one_game.append(attacker.HEU[attacker.chosen_strategy])
        att_HEU_one_game = np.array(att_HEU_one_game)
        self.att_HEU_history.append(att_HEU_one_game)
        self.def_HEU_history.append(
            self.defender.HEU[self.defender.chosen_strategy_list])

        # AHEU/DHEU per Strategy
        AHEU_per_game = {}
        for attacker in self.attacker_list:
            AHEU_per_game[attacker.attacker_ID] = attacker.HEU
        self.AHEU_per_Strategy_History.append(AHEU_per_game)
        self.DHEU_per_Strategy_History.append(self.defender.HEU)

        # Att/Def Strategy
        att_strat_one_game = []
        for attacker in self.attacker_list:
            att_strat_one_game.append(attacker.chosen_strategy)
        self.att_strategy_counter.append(att_strat_one_game)
        self.def_strategy_counter.append(self.defender.chosen_strategy_list)

        # FP & TP for ROC curve
        self.FPR_history.append(1 - self.TN /
                                (self.TN + self.FP))  # FPR using preset value
        self.TPR_history.append(1 - self.FN / (self.FN + self.TP))

        # Att/Def Cost
        att_cost_in_one_game = []
        for attacker in self.attacker_list:
            att_cost_in_one_game.append(attacker.strat_cost[attacker.chosen_strategy])
        att_cost_in_one_game = np.array(att_cost_in_one_game)
        self.att_cost_history.append(att_cost_in_one_game)
        # self.att_cost_history.append(
        #     self.attacker.strat_cost[self.attacker.chosen_strategy])
        self.def_cost_history.append(
            self.defender.strat_cost[self.defender.chosen_strategy_list])
        # Criticality
        criti_list = (np.array(
            list(
                nx.get_node_attributes(self.graph.network,
                                       "criticality").values())) * 1000).astype(int)
        for value in criti_list:
            self.criticality_hisotry[value] += 1

        # attacker number
        self.att_number.append(len(self.attacker_list))

        # attacker CKC
        self.att_CKC.append(get_CKC_list(self.attacker_list))

        # inside attacker counter
        one_game_counter = [0,0]
        for attacker in self.attacker_list:
            one_game_counter[0] += 1
            if attacker.in_system_time > 1:
                one_game_counter[1] += 1
        # print(f"total vs inside attacker: {one_game_counter}")
        self.number_of_inside_attacker.append(one_game_counter)




        # # EU_C & EU_CMS
        # self.att_EU_C = np.vstack((self.att_EU_C, self.attacker.EU_C))
        # self.att_EU_CMS = np.vstack((self.att_EU_CMS, self.attacker.EU_CMS))
        # self.def_EU_C = np.vstack((self.def_EU_C, self.defender.EU_C))
        # self.def_EU_CMS = np.vstack((self.def_EU_CMS, self.defender.EU_CMS))
        # # attacker/defender impact
        for attacker in self.attacker_list:
            self.att_impact = np.vstack(
                (self.att_impact, attacker.impact_record))
        self.def_impact = np.vstack(
            (self.def_impact, self.defender.impact_record))
        # # HEU in DD IPI
        # self.att_HEU_DD_IPI = np.vstack(
        #     (self.att_HEU_DD_IPI, self.attacker.HEU))
        # self.def_HEU_DD_IPI = np.vstack(
        #     (self.def_HEU_DD_IPI, self.defender.HEU))

# In[ ]:


# In[ ]:


# In[ ]:
