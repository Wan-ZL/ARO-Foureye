#!/usr/bin/env python
# coding: utf-8


# # solve Jupyter runtime error
# import nest_asyncio
# nest_asyncio.apply()


import concurrent.futures
import multiprocessing
import os

import matplotlib.pyplot as plt
from networkx import nx
import numpy as np
from itertools import count
import random
import math
import copy
import time
import pickle

from graph_function import *
from attacker_function import *
from defender_function import *
from main import display


class game_class:
    def __init__(self, simulation_id, DD_using, uncertain_scheme, web_data_upper_vul, Iot_upper_vul, Th_risk, _lambda,
                 mu, SF_thres_1, SF_thres_2, att_detect_UpBod):
        self.lifetime = 1
        self.CKC_number = 6
        self.strategy_number = 8
        self.DD_using = DD_using
        self.graph = graph_class(web_data_upper_vul, Iot_upper_vul)
        self.uncertain_scheme = uncertain_scheme
        self.att_detect_UpBod = att_detect_UpBod
        self.attacker = attacker_class(self, self.uncertain_scheme, self.att_detect_UpBod)
        self.attacker_number = 1
        self.defender = defender_class(self, self.uncertain_scheme)
        self.game_over = False
        self.FN = 10  # False Negative for Beta distribution
        self.TP = 90  # True Positive
        self.TN = 99
        self.FP = 1
        self.rewire_network = 0.01
        # for Experiment Result ⬇️
        self.def_uncertainty_history = []
        self.att_uncertainty_history = []
        self.pre_attacker_number = 0
        self.att_HEU_history = []
        self.def_HEU_history = []
        self.att_strategy_counter = []
        self.def_strategy_counter = []
        self.FPR_history = []
        self.TPR_history = []
        self.att_cost_history = []
        self.def_cost_history = []
        self.def_per_strat_cost = np.zeros((1,8))
        self.def_succ_counter = np.zeros((6,8))
        self.def_fail_counter = np.zeros((6,8))
        self.criticality_hisotry = np.zeros(100000)  # np.zeros(10000)
        self.evict_reason_history = np.zeros(2)
        self.SysFail = [False] * 3
        self.att_EU_C = np.zeros(8)
        self.att_EU_CMS = np.zeros(8)
        self.def_EU_C = np.zeros(8)
        self.def_EU_CMS = np.zeros(8)
        self.att_impact = np.zeros(8)
        self.def_impact = np.zeros(8)
        self.att_HEU_DD_IPI = np.zeros(8)
        self.def_HEU_DD_IPI = np.zeros(8)
        self.NIDS_eviction = np.zeros(4)  # [# of bad, # of good]
        self.NIDS_Th_risk = Th_risk
        self._lambda = _lambda
        self.mu = mu
        self.SF_thres_1 = SF_thres_1
        self.SF_thres_2 = SF_thres_2
        self.hitting_result = []

    def attacker_round(self, simulation_id):
        if display: print(f"attacker location{self.attacker.location}")
        if self.game_over:
            print(f"Sim {simulation_id} GAME OVER")
            return

        self.attacker.observe_opponent(self.defender.CKC_position,
                                       self.defender.chosen_strategy)

        self.attacker.choose_strategy(self.defender.strategy_number,
                                      self.defender.strat_cost,
                                      self.defender.impact_record)
        if display:
            print(f"attacker choose: {self.attacker.chosen_strategy + 1}")
        attack_result = self.attacker.execute_strategy(
            self.graph.network, self.defender.network, self.defender.P_fake,
            self.attacker.detect_prob)
        self.attacker.update_attribute(self.defender.dec, self._lambda)
        self.graph.update_graph(self.defender.network, self.attacker.network)
        if attack_result:
            self.def_succ_counter[self.attacker.CKC_position, self.defender.chosen_strategy] += 1
        else:
            self.def_fail_counter[self.attacker.CKC_position, self.defender.chosen_strategy] += 1
        if attack_result:
            if (self.attacker.chosen_strategy == 0
                    and self.attacker.CKC_position != 0):
                pass  # This avoid inside attacker increase stage when Strategy 1 success
            else:
                self.attacker.next_stage()
        else:
            self.attacker.random_moving()
            if display:
                print(f"attacker move, new location: {self.attacker.location}")

        return attack_result

    def defender_round(self):
        self.defender.observe_opponent(self.attacker.impact_record,
                                       self.attacker.CKC_position,
                                       self.attacker.chosen_strategy)
        result = self.defender.decide_CKC_posi(self.attacker.detect_prob,
                                               self.attacker.CKC_position)
        if result:
            if display:
                print("defender guess CKC correct")
        else:
            if display:
                print("defender guess CKC wrong")

        self.defender.choose_strategy(self.attacker.chosen_strategy,
                                      self.attacker.strategy_number,
                                      self.attacker.strat_cost,
                                      self.attacker.impact_record)
        if display:
            print(f"defender choose: {self.defender.chosen_strategy + 1}")
        success = self.defender.execute_strategy(self.attacker.network,
                                                 self.attacker.detect_prob,
                                                 self.graph,
                                                 self.FN / (self.TP + self.FN),
                                                 self.FP / (self.TN + self.FP), self.NIDS_eviction)
        self.defender.update_attribute(self.attacker.detect_prob, self.mu, self.attacker.impact_record)
        self.graph.update_graph(self.defender.network, self.attacker.network)

    def NIDS_detect(self):
        # Warning: False Positive evict too many nodes
        # false negative rate
        false_neg_prob = self.FN / (self.TP + self.FN)
        false_pos_prob = self.FP / (self.TN + self.FP)
        Th_risk = self.NIDS_Th_risk

        #         for index in self.graph.network.nodes:
        all_nodes = list(self.graph.network.nodes(data=False))
        experiment_index_record = 0
        for index in all_nodes:
            if is_node_evicted(self.graph.network,
                               index):  # ignore evicted node for saving time
                continue

            # detect is node compromised
            node_is_compromised = False
            if self.graph.network.has_node(index):
                if self.graph.network.nodes[index]["compromised_status"]:
                    if random.random() > false_neg_prob:
                        node_is_compromised = True
                        self.defender.network.nodes[index]["compromised_status"] = True
                        experiment_index_record = 0
                    else:
                        if display: print("False Negative to compromised node")
                else:
                    if random.random() < false_pos_prob:
                        if display: print("False Positive to good node")
                        node_is_compromised = True
                        self.defender.network.nodes[index]["compromised_status"] = True
                        experiment_index_record = 1

            if node_is_compromised:
                # No-DD means NIDS doesn't remain attacker in system
                if not self.DD_using:
                    if display: print(f"Evict node {index}, No DD using")
                    evict_a_node(index, self.graph.network,
                                 self.defender.network, self.attacker.network)
                    self.NIDS_eviction[experiment_index_record] += 1
                    continue
                if self.graph.network.has_node(index):
                    if self.graph.network.nodes[index]["criticality"] > Th_risk:
                        if display:
                            print(f"Evict node {index}, criticality > Th_risk")
                        evict_a_node(index, self.graph.network,
                                     self.defender.network,
                                     self.attacker.network)
                        self.NIDS_eviction[experiment_index_record] += 1
                        continue
                    else:
                        if is_system_fail(self.graph, [None], self.SF_thres_1, self.SF_thres_2):
                            if display:
                                print(
                                    f"Evict node {index}, compromise cause SF")
                            evict_a_node(index, self.graph.network,
                                         self.defender.network,
                                         self.attacker.network)
                            self.NIDS_eviction[experiment_index_record] += 1

    def update_graph(self):
        self.graph.update_graph()
        self.attacker.update_graph()
        self.defender.update_graph()

    def prepare_for_next_game(self):
        self.lifetime += 1

        # Beta distribution
        if self.graph.using_honeynet:
            self.TP += 5
            self.TN += 5

        else:
            if self.defender.chosen_strategy == 4 or self.defender.chosen_strategy == 5 or self.defender.chosen_strategy == 6 or self.defender.chosen_strategy == 7:
                self.TP += 5
                self.TN += 5

        # rewire graph
        rewire_network(self.graph.network, self.attacker.network,
                       self.defender.network, self.rewire_network)

        # reconnect non-evicted node to server or databse
        node_reconnect(self.graph.network, self.attacker.network,
                       self.defender.network, self.graph.connect_prob)

        # update defender impact
        self.defender.impact_record[
            self.defender.chosen_strategy] = 1 - self.attacker.impact_record[
            self.attacker.chosen_strategy]

        # clean honeypot after each game
        if self.graph.using_honeynet:
            clean_honeynet(self.graph.network, self.attacker.network,
                           self.defender.network)
            self.graph.using_honeynet = False

        # remove honeypot in comrpomised list
        for index in self.attacker.compromised_nodes:
            if not self.graph.network.has_node(index):
                self.attacker.compromised_nodes.remove(index)
        # remove honeypot in collection list
        for index in self.attacker.collection_list:
            if not self.graph.network.has_node(index):
                self.attacker.collection_list.remove(index)

    def new_attacker(self, simulation_id):
        self.attacker_number += 1
        print(
            f"\033[93m Sim {simulation_id} Creating attacker #{self.attacker_number} \033[0m"
        )
        # new attacker
        self.attacker = attacker_class(self, self.uncertain_scheme, self.att_detect_UpBod)
        # reset defender
        self.defender.reset_attribute(self.attacker.impact_record,
                                      self.CKC_number)

    def experiment_saving(self):
        self.def_uncertainty_history.append(self.defender.uncertainty)
        self.att_uncertainty_history.append(self.attacker.uncertainty)

        # Att/Def HEU
        self.att_HEU_history.append(
            self.attacker.HEU[self.attacker.chosen_strategy])
        self.def_HEU_history.append(
            self.defender.HEU[self.defender.chosen_strategy])
        # Att/Def Strategy
        self.att_strategy_counter.append(self.attacker.chosen_strategy)
        self.def_strategy_counter.append(self.defender.chosen_strategy)
        # FP & TP for ROC curve
        self.FPR_history.append(1 - self.TN /
                                (self.TN + self.FP))  # FPR using preset value
        self.TPR_history.append(1 - self.FN / (self.FN + self.TP))
        # Att/Def Cost
        self.att_cost_history.append(
            self.attacker.strat_cost[self.attacker.chosen_strategy])
        self.def_cost_history.append(
            self.defender.strat_cost[self.defender.chosen_strategy])
        def_cost_temp = np.zeros(8)
        def_cost_temp[self.defender.chosen_strategy] = self.defender.strat_cost[self.defender.chosen_strategy]

        self.def_per_strat_cost = np.append(self.def_per_strat_cost, np.reshape(def_cost_temp, (1, -1)), axis=0)


        # Criticality
        criti_list = (np.array(
            list(
                nx.get_node_attributes(self.graph.network,
                                       "criticality").values())) *
                      1000).astype(int)
        for value in criti_list:
            self.criticality_hisotry[value] += 1
        # EU_C & EU_CMS
        self.att_EU_C = np.vstack((self.att_EU_C, self.attacker.EU_C))
        self.att_EU_CMS = np.vstack((self.att_EU_CMS, self.attacker.EU_CMS))
        self.def_EU_C = np.vstack((self.def_EU_C, self.defender.EU_C))
        self.def_EU_CMS = np.vstack((self.def_EU_CMS, self.defender.EU_CMS))
        # attacker/defender impact
        self.att_impact = np.vstack(
            (self.att_impact, self.attacker.impact_record))
        self.def_impact = np.vstack(
            (self.def_impact, self.defender.impact_record))
        # HEU in DD IPI
        self.att_HEU_DD_IPI = np.vstack(
            (self.att_HEU_DD_IPI, self.attacker.HEU))
        self.def_HEU_DD_IPI = np.vstack(
            (self.def_HEU_DD_IPI, self.defender.HEU))

        # Hitting Ratio
        hit = False
        att_AHEU_str_index = random.choice(np.where(self.attacker.AHEU == max(self.attacker.AHEU))[0])
        att_DHEU_str_index = random.choice(np.where(self.attacker.att_guess_DHEU == max(self.attacker.att_guess_DHEU))[0])

        def_AHEU_str_index = random.choice(np.where(self.defender.def_guess_AHEU == max(self.defender.def_guess_AHEU))[0])
        # def_DHEU_str_index = random.choice(np.where(attacker.defender_HEU == max(attacker.defender_HEU))[0])
        def_DHEU_str_index = random.choice(np.where(self.defender.DHEU == max(self.defender.DHEU))[0])
        if att_AHEU_str_index == def_AHEU_str_index and att_DHEU_str_index == def_DHEU_str_index:
            self.hitting_result.append(True)
        else:
            self.hitting_result.append(False)


def game_start(simulation_id=0,
               DD_using=True,
               uncertain_scheme=True,
               web_data_upper_vul=7,
               Iot_upper_vul=5, Th_risk=0.3, _lambda=1, mu=8, SF_thres_1=1 / 3, SF_thres_2=1 / 2, att_detect_UpBod=0.5):
    print(
        f"Start Simulation {simulation_id}, DD_using={DD_using}, uncertain_scheme={uncertain_scheme}, web_data_upper_vul={web_data_upper_vul}, Iot_upper_vul={Iot_upper_vul}"
    )
    np.seterr(divide='ignore',
              invalid='ignore')  # for remove divide zero warning

    game_continue = True

    game = game_class(simulation_id, DD_using, uncertain_scheme,
                      web_data_upper_vul, Iot_upper_vul, Th_risk, _lambda, mu, SF_thres_1, SF_thres_2, att_detect_UpBod)

    while (not game.game_over):
        print(game.lifetime)
        if display:
            print(f"attacker CKC: {game.attacker.CKC_position + 1}")

        #         print(game.lifetime)
        game.defender_round()
        attack_result = game.attacker_round(simulation_id)
        game.experiment_saving()
        game.NIDS_detect()
        att_outside = False
        if game.attacker.location is not None:
            att_outside = is_node_evicted(game.graph.network,
                                          game.attacker.location)

        # Decide whether to create new attacker
        if att_outside:
            game.evict_reason_history[0] += 1
        # check is attacker in honeypot
        att_in_honeypot = False
        if game.attacker.location is not None:
            if game.graph.network.has_node(game.attacker.location):
                if game.graph.network.nodes[
                    game.attacker.location]["type"] == 3:
                    att_in_honeypot = True
                    game.evict_reason_history[1] += 1
        reason_box = [None]
        if is_system_fail(game.graph, reason_box, SF_thres_1, SF_thres_2):
            print(f"Sim {simulation_id} SYSTEM FAIL \U0001F480")
            print(f"Sim {simulation_id} GAME OVER")
            game.game_over = True
            game.SysFail[reason_box[0]] = True
        game.prepare_for_next_game()
        data_exfil_succ = False
        if attack_result:
            if game.attacker.chosen_strategy == 7:
                data_exfil_succ = True
                print("Strategy 8 win !!!")
        if att_outside or att_in_honeypot or data_exfil_succ:
            game.new_attacker(simulation_id)

        # if all-3 node evicted, End simulation
        all_evict_mark = list(
            nx.get_node_attributes(game.graph.network,
                                   "evicted_mark").values())
        if sum(all_evict_mark) >= len(all_evict_mark) - 3:
            print(f"Sim {simulation_id} All node evicted")
            game.SysFail[0] = True
            game.game_over = True

    if display: draw_graph(game.attacker.network)
    if display: draw_graph(game.graph.network)
    return game


def run_sumulation_fixed_setting(current_scheme, DD_using, uncertain_scheme,
                                 simulation_time):
    #     simulation_time = 100

    #     start = time.perf_counter()

    def_uncertainty_all_result = {}
    att_uncertainty_all_result = {}
    Time_to_SF_all_result = {}
    att_HEU_all_result = {}
    def_HEU_all_result = {}
    att_strategy_count_result = {}
    def_strategy_count_result = {}
    FPR_all_result = {}
    TPR_all_result = {}
    att_cost_all_result = {}
    def_cost_all_result = {}
    criticality_all_result = {}
    evict_reason_all_result = {}
    SysFail_reason = [0] * 3
    att_EU_C_all_result = {}
    att_EU_CMS_all_result = {}
    def_EU_C_all_result = {}
    def_EU_CMS_all_result = {}
    att_impact_all_result = {}
    def_impact_all_result = {}
    att_HEU_DD_IPI_all_result = {}
    def_HEU_DD_IPI_all_result = {}
    NIDS_eviction_all_result = {}
    hitting_probability_all_result = {}
    def_succ_counter_all_result = {}
    def_fail_counter_all_result = {}
    cost_per_strat_allresult = {}

    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for i in range(simulation_time):
            future = executor.submit(game_start, i, DD_using,
                                     uncertain_scheme)  # scheme change here
            results.append(future)

        index = 0
        for future in results:
            # New Attacker
            Time_to_SF_all_result[index] = future.result().lifetime
            # HEU
            att_HEU_all_result[index] = future.result().att_HEU_history
            def_HEU_all_result[index] = future.result().def_HEU_history
            # Strategy Counter
            att_strategy_count_result[index] = future.result(
            ).att_strategy_counter
            def_strategy_count_result[index] = future.result(
            ).def_strategy_counter
            # Uncertainty
            def_uncertainty_all_result[index] = future.result(
            ).def_uncertainty_history
            att_uncertainty_all_result[index] = future.result(
            ).att_uncertainty_history
            # TPR & FPR
            FPR_all_result[index] = future.result().FPR_history
            TPR_all_result[index] = future.result().TPR_history
            # Cost
            att_cost_all_result[index] = future.result().att_cost_history
            def_cost_all_result[index] = future.result().def_cost_history
            # Criticality
            criticality_all_result[index] = future.result().criticality_hisotry
            # Evict attacker reason
            evict_reason_all_result[index] = future.result(
            ).evict_reason_history
            # System Fail reason
            if future.result().SysFail[0]:
                SysFail_reason[0] += 1  # [att_strat, system_fail]
            elif future.result().SysFail[1]:
                SysFail_reason[1] += 1
            elif future.result().SysFail[2]:
                SysFail_reason[2] += 1
            # EU_C & EU_CMS
            att_EU_C_all_result[index] = np.delete(future.result().att_EU_C, 0,
                                                   0)
            att_EU_CMS_all_result[index] = np.delete(
                future.result().att_EU_CMS, 0, 0)
            def_EU_C_all_result[index] = np.delete(future.result().def_EU_C, 0,
                                                   0)
            def_EU_CMS_all_result[index] = np.delete(
                future.result().def_EU_CMS, 0, 0)
            # attacker & defender impact
            att_impact_all_result[index] = np.delete(
                future.result().att_impact, 0, 0)
            def_impact_all_result[index] = np.delete(
                future.result().def_impact, 0, 0)
            # HEU in DD IPI
            att_HEU_DD_IPI_all_result[index] = np.delete(
                future.result().att_HEU_DD_IPI, 0, 0)
            def_HEU_DD_IPI_all_result[index] = np.delete(
                future.result().def_HEU_DD_IPI, 0, 0)
            # NIDS evict Bad or Good
            NIDS_eviction_all_result[index] = future.result().NIDS_eviction
            # hitting probability for Hypergame Nash Equilibrium
            hitting_probability_all_result[index] = future.result().hitting_result
            # defender success/fail counter
            def_succ_counter_all_result[index] = future.result().def_succ_counter
            def_fail_counter_all_result[index] = future.result().def_fail_counter
            # defender cost per strategy
            cost_per_strat_allresult[index] = future.result().def_per_strat_cost[1:]

            index += 1

    # SAVE to FILE (need to create directory manually)
    # history of when new attacker created
    os.makedirs("data/" + current_scheme, exist_ok=True)
    the_file = open("data/" + current_scheme + "/Time_to_SF.pkl", "wb+")
    pickle.dump(Time_to_SF_all_result, the_file)
    the_file.close()

    # HEU
    os.makedirs("data/" + current_scheme + "/R1", exist_ok=True)
    the_file = open("data/" + current_scheme + "/R1/att_HEU.pkl", "wb+")
    pickle.dump(att_HEU_all_result, the_file)
    the_file.close()
    the_file = open("data/" + current_scheme + "/R1/def_HEU.pkl", "wb+")
    pickle.dump(def_HEU_all_result, the_file)
    the_file.close()

    # Strategy Counter
    os.makedirs("data/" + current_scheme + "/R2", exist_ok=True)
    the_file = open("data/" + current_scheme + "/R2/att_strategy_counter.pkl",
                    "wb+")
    pickle.dump(att_strategy_count_result, the_file)
    the_file.close()
    the_file = open("data/" + current_scheme + "/R2/def_strategy_counter.pkl",
                    "wb+")
    pickle.dump(def_strategy_count_result, the_file)
    the_file.close()

    # uncertainty
    os.makedirs("data/" + current_scheme + "/R3", exist_ok=True)
    the_file = open("data/" + current_scheme + "/R3/defender_uncertainty.pkl",
                    "wb+")
    pickle.dump(def_uncertainty_all_result, the_file)
    the_file.close()
    the_file = open("data/" + current_scheme + "/R3/attacker_uncertainty.pkl",
                    "wb+")
    pickle.dump(att_uncertainty_all_result, the_file)
    the_file.close()

    # TPR & FPR
    os.makedirs("data/" + current_scheme + "/R4", exist_ok=True)
    the_file = open("data/" + current_scheme + "/R4/FPR.pkl", "wb+")
    pickle.dump(FPR_all_result, the_file)
    the_file.close()
    the_file = open("data/" + current_scheme + "/R4/TPR.pkl", "wb+")
    pickle.dump(TPR_all_result, the_file)
    the_file.close()

    # Cost
    os.makedirs("data/" + current_scheme + "/R6", exist_ok=True)
    the_file = open("data/" + current_scheme + "/R6/att_cost.pkl", "wb+")
    pickle.dump(att_cost_all_result, the_file)
    the_file.close()
    the_file = open("data/" + current_scheme + "/R6/def_cost.pkl", "wb+")
    pickle.dump(def_cost_all_result, the_file)
    the_file.close()

    # Criticality
    # os.makedirs("data/" + current_scheme + "/R_self_1", exist_ok=True)
    # the_file = open("data/" + current_scheme + "/R_self_1/criticality.pkl",
    #                 "wb+")
    # pickle.dump(criticality_all_result, the_file)
    # the_file.close()

    # Evict attacker reason
    os.makedirs("data/" + current_scheme + "/R_self_2", exist_ok=True)
    the_file = open("data/" + current_scheme + "/R_self_2/evict_reason.pkl",
                    "wb+")
    pickle.dump(evict_reason_all_result, the_file)
    the_file.close()

    # System Failure reason
    os.makedirs("data/" + current_scheme + "/R_self_3", exist_ok=True)
    the_file = open("data/" + current_scheme + "/R_self_3/system_fail.pkl",
                    "wb+")
    pickle.dump(SysFail_reason, the_file)
    the_file.close()

    # EU_C & EU_CMS
    os.makedirs("data/" + current_scheme + "/R_self_4", exist_ok=True)
    the_file = open("data/" + current_scheme + "/R_self_4/att_EU_C.pkl", "wb+")
    pickle.dump(att_EU_C_all_result, the_file)
    the_file.close()
    the_file = open("data/" + current_scheme + "/R_self_4/att_EU_CMS.pkl",
                    "wb+")
    pickle.dump(att_EU_CMS_all_result, the_file)
    the_file.close()
    the_file = open("data/" + current_scheme + "/R_self_4/def_EU_C.pkl", "wb+")
    pickle.dump(def_EU_C_all_result, the_file)
    the_file.close()
    the_file = open("data/" + current_scheme + "/R_self_4/def_EU_CMS.pkl",
                    "wb+")
    pickle.dump(def_EU_CMS_all_result, the_file)
    the_file.close()

    # attacker & defender impact
    the_file = open("data/" + current_scheme + "/R_self_4/att_impact.pkl",
                    "wb+")
    pickle.dump(att_impact_all_result, the_file)
    the_file.close()
    the_file = open("data/" + current_scheme + "/R_self_4/def_impact.pkl",
                    "wb+")
    pickle.dump(def_impact_all_result, the_file)
    the_file.close()

    # HEU in DD IPI
    the_file = open("data/" + current_scheme + "/R_self_4/att_HEU_DD_IPI.pkl",
                    "wb+")
    pickle.dump(att_HEU_DD_IPI_all_result, the_file)
    the_file.close()
    the_file = open("data/" + current_scheme + "/R_self_4/def_HEU_DD_IPI.pkl",
                    "wb+")
    pickle.dump(def_HEU_DD_IPI_all_result, the_file)
    the_file.close()

    # NIDS evict good or bad
    the_file = open("data/" + current_scheme + "/R_self_4/NIDS_eviction.pkl",
                    "wb+")
    pickle.dump(NIDS_eviction_all_result, the_file)
    the_file.close()

    # Hitting Probability
    the_file = open("data/" + current_scheme + "/R_self_4/hitting_probability.pkl", "wb+")
    pickle.dump(hitting_probability_all_result, the_file)
    the_file.close()

    # defender strategy success or failure
    os.makedirs("data/" + current_scheme + "/R6", exist_ok=True)
    the_file = open("data/" + current_scheme + "/R6/def_succ_counter.pkl", "wb+")
    pickle.dump(def_succ_counter_all_result, the_file)
    the_file.close()
    the_file = open("data/" + current_scheme + "/R6/def_fail_counter.pkl", "wb+")
    pickle.dump(def_fail_counter_all_result, the_file)
    the_file.close()

    # defender cost per strategy
    os.makedirs("data/" + current_scheme + "/R6", exist_ok=True)
    the_file = open("data/" + current_scheme + "/R6/def_cost_per_strat.pkl", "wb+")
    pickle.dump(cost_per_strat_allresult, the_file)
    the_file.close()


def run_sumulation_group_varying_vul(current_scheme, DD_using, uncertain_scheme,
                                     simulation_time):
    vul_range = {}
    MTTSF_all_result = np.zeros(5)
    att_cost_all_result = np.zeros(5)
    def_cost_all_result = np.zeros(5)
    att_HEU_all_result = np.zeros(5)
    def_HEU_all_result = np.zeros(5)
    att_uncertainty_all_result = np.zeros(5)
    def_uncertainty_all_result = np.zeros(5)
    FPR_all_result = np.zeros(5)
    TPR_all_result = np.zeros(5)

    # web_data_SoftVul_range = range(3,7+1)
    # IoT_SoftVul_range = range(1,5+1)
    web_data_SoftVul_range = np.array(range(1, 5 + 1)) * 2
    IoT_SoftVul_range = np.array(range(1, 5 + 1)) * 2
    vul_range[0] = web_data_SoftVul_range
    vul_range[1] = IoT_SoftVul_range

    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for vul_index in range(5):
            particular_vul_result = []
            for i in range(simulation_time):
                future = executor.submit(
                    game_start, i, DD_using, uncertain_scheme,
                    web_data_upper_vul=web_data_SoftVul_range[vul_index],
                    Iot_upper_vul=IoT_SoftVul_range[vul_index])  # scheme change here
                particular_vul_result.append(future)
            results.append(particular_vul_result)

        index = 0
        for particular_vul_result in results:
            total_time_for_all_sim = 0
            for future in particular_vul_result:
                # change web server and database vul
                # MTTSF
                MTTSF_all_result[index] += future.result().lifetime
                # Cost
                att_cost_all_result[index] += sum(
                    future.result().att_cost_history) / len(
                    future.result().att_cost_history)
                def_cost_all_result[index] += sum(
                    future.result().def_cost_history) / len(
                    future.result().def_cost_history)
                # HEU
                att_HEU_all_result[index] += sum(
                    future.result().att_HEU_history) / len(
                    future.result().att_HEU_history)
                def_HEU_all_result[index] += sum(
                    future.result().def_HEU_history) / len(
                    future.result().def_HEU_history)
                # Uncertainty
                att_uncertainty_all_result[index] += sum(
                    future.result().att_uncertainty_history) / len(
                    future.result().att_uncertainty_history)
                def_uncertainty_all_result[index] += sum(
                    future.result().def_uncertainty_history) / len(
                    future.result().def_uncertainty_history)
                # FPR & TPR
                FPR_all_result[index] += sum(
                    future.result().FPR_history) / len(
                    future.result().FPR_history)
                TPR_all_result[index] += sum(
                    future.result().TPR_history) / len(
                    future.result().TPR_history)
                total_time_for_all_sim += 1

            att_cost_all_result[
                index] = att_cost_all_result[index] / total_time_for_all_sim
            def_cost_all_result[
                index] = def_cost_all_result[index] / total_time_for_all_sim
            att_HEU_all_result[
                index] = att_HEU_all_result[index] / total_time_for_all_sim
            def_HEU_all_result[
                index] = def_HEU_all_result[index] / total_time_for_all_sim
            att_uncertainty_all_result[index] = att_uncertainty_all_result[
                                                    index] / total_time_for_all_sim
            def_uncertainty_all_result[index] = def_uncertainty_all_result[
                                                    index] / total_time_for_all_sim
            FPR_all_result[
                index] = FPR_all_result[index] / total_time_for_all_sim
            TPR_all_result[
                index] = TPR_all_result[index] / total_time_for_all_sim
            MTTSF_all_result[index] = MTTSF_all_result[index] / simulation_time
            index += 1

    # SAVE to FILE (need to create directory manually)
    # Vul range
    os.makedirs("data/" + current_scheme + "/VUB", exist_ok=True)
    the_file = open("data/" + current_scheme + "/VUB/Range.pkl", "wb+")
    pickle.dump(vul_range, the_file)
    the_file.close()
    # MTTSF
    os.makedirs("data/" + current_scheme + "/VUB", exist_ok=True)
    the_file = open("data/" + current_scheme + "/VUB/MTTSF.pkl", "wb+")
    pickle.dump(MTTSF_all_result, the_file)
    the_file.close()

    # Cost
    os.makedirs("data/" + current_scheme + "/VUB", exist_ok=True)
    the_file = open("data/" + current_scheme + "/VUB/att_cost.pkl", "wb+")
    pickle.dump(att_cost_all_result, the_file)
    the_file.close()
    the_file = open("data/" + current_scheme + "/VUB/def_cost.pkl", "wb+")
    pickle.dump(def_cost_all_result, the_file)
    the_file.close()

    # HEU
    os.makedirs("data/" + current_scheme + "/VUB", exist_ok=True)
    the_file = open("data/" + current_scheme + "/VUB/att_HEU.pkl", "wb+")
    pickle.dump(att_HEU_all_result, the_file)
    the_file.close()
    the_file = open("data/" + current_scheme + "/VUB/def_HEU.pkl", "wb+")
    pickle.dump(def_HEU_all_result, the_file)
    the_file.close()

    # Uncertainty
    os.makedirs("data/" + current_scheme + "/VUB", exist_ok=True)
    the_file = open("data/" + current_scheme + "/VUB/att_uncertainty.pkl",
                    "wb+")
    pickle.dump(att_uncertainty_all_result, the_file)
    the_file.close()
    the_file = open("data/" + current_scheme + "/VUB/def_uncertainty.pkl",
                    "wb+")
    pickle.dump(def_uncertainty_all_result, the_file)
    the_file.close()

    # FPR & TPR
    os.makedirs("data/" + current_scheme + "/VUB", exist_ok=True)
    the_file = open("data/" + current_scheme + "/VUB/FPR.pkl", "wb+")
    pickle.dump(FPR_all_result, the_file)
    the_file.close()
    the_file = open("data/" + current_scheme + "/VUB/TPR.pkl", "wb+")
    pickle.dump(TPR_all_result, the_file)
    the_file.close()


# def run_sumulation_group_varying_Th_risk(current_scheme, DD_using, uncertain_scheme,
#                                          simulation_time):
#     varying_range = [0.1, 0.2, 0.3, 0.4, 0.5]
#
#     MTTSF_all_result = np.zeros(len(varying_range))
#     FPR_all_result = np.zeros(len(varying_range))
#     TPR_all_result = np.zeros(len(varying_range))
#
#     results = []
#     with concurrent.futures.ProcessPoolExecutor() as executor:
#         for vary_index in range(len(varying_range)):
#             particular_vul_result = []
#             for i in range(simulation_time):
#                 future = executor.submit(
#                     game_start, i, DD_using, uncertain_scheme, Th_risk=varying_range[vary_index])  # scheme change here
#                 particular_vul_result.append(future)
#             results.append(particular_vul_result)
#
#         index = 0
#         for particular_vul_result in results:
#             total_time_for_all_sim = 0
#             for future in particular_vul_result:
#                 # change web server and database vul
#                 # MTTSF
#                 MTTSF_all_result[index] += future.result().lifetime
#                 # FPR & TPR
#                 FPR_all_result[index] += sum(
#                     future.result().FPR_history) / len(
#                     future.result().FPR_history)
#                 TPR_all_result[index] += sum(
#                     future.result().TPR_history) / len(
#                     future.result().TPR_history)
#                 total_time_for_all_sim += 1
#
#             FPR_all_result[
#                 index] = FPR_all_result[index] / total_time_for_all_sim
#             TPR_all_result[
#                 index] = TPR_all_result[index] / total_time_for_all_sim
#             MTTSF_all_result[index] = MTTSF_all_result[index] / simulation_time
#             index += 1
#
#     # SAVE to FILE (need to create directory manually)
#     # vary range
#     os.makedirs("data/" + current_scheme + "/Th_risk", exist_ok=True)
#     the_file = open("data/" + current_scheme + "/Th_risk/Range.pkl", "wb+")
#     pickle.dump(varying_range, the_file)
#     the_file.close()
#     # MTTSF
#     os.makedirs("data/" + current_scheme + "/Th_risk", exist_ok=True)
#     the_file = open("data/" + current_scheme + "/Th_risk/MTTSF.pkl", "wb+")
#     pickle.dump(MTTSF_all_result, the_file)
#     the_file.close()
#
#     # FPR & TPR
#     os.makedirs("data/" + current_scheme + "/Th_risk", exist_ok=True)
#     the_file = open("data/" + current_scheme + "/Th_risk/FPR.pkl", "wb+")
#     pickle.dump(FPR_all_result, the_file)
#     the_file.close()
#     the_file = open("data/" + current_scheme + "/Th_risk/TPR.pkl", "wb+")
#     pickle.dump(TPR_all_result, the_file)
#     the_file.close()


# def run_sumulation_group_varying_lambda(current_scheme, DD_using, uncertain_scheme,
#                                         simulation_time):
#     varying_range = [0.6, 0.7, 0.8, 0.9, 1]
#
#     MTTSF_all_result = np.zeros(len(varying_range))
#     FPR_all_result = np.zeros(len(varying_range))
#     TPR_all_result = np.zeros(len(varying_range))
#
#     results = []
#     with concurrent.futures.ProcessPoolExecutor() as executor:
#         for vary_index in range(len(varying_range)):
#             particular_vul_result = []
#             for i in range(simulation_time):
#                 future = executor.submit(
#                     game_start, i, DD_using, uncertain_scheme, _lambda=varying_range[vary_index])  # scheme change here
#                 particular_vul_result.append(future)
#             results.append(particular_vul_result)
#
#         index = 0
#         for particular_vul_result in results:
#             total_time_for_all_sim = 0
#             for future in particular_vul_result:
#                 # change web server and database vul
#                 # MTTSF
#                 MTTSF_all_result[index] += future.result().lifetime
#                 # FPR & TPR
#                 FPR_all_result[index] += sum(
#                     future.result().FPR_history) / len(
#                     future.result().FPR_history)
#                 TPR_all_result[index] += sum(
#                     future.result().TPR_history) / len(
#                     future.result().TPR_history)
#                 total_time_for_all_sim += 1
#
#             FPR_all_result[
#                 index] = FPR_all_result[index] / total_time_for_all_sim
#             TPR_all_result[
#                 index] = TPR_all_result[index] / total_time_for_all_sim
#             MTTSF_all_result[index] = MTTSF_all_result[index] / simulation_time
#             index += 1
#
#     # SAVE to FILE (need to create directory manually)
#     # vary range
#     os.makedirs("data/" + current_scheme + "/_lambda", exist_ok=True)
#     the_file = open("data/" + current_scheme + "/_lambda/Range.pkl", "wb+")
#     pickle.dump(varying_range, the_file)
#     the_file.close()
#     # MTTSF
#     os.makedirs("data/" + current_scheme + "/_lambda", exist_ok=True)
#     the_file = open("data/" + current_scheme + "/_lambda/MTTSF.pkl", "wb+")
#     pickle.dump(MTTSF_all_result, the_file)
#     the_file.close()
#
#     # FPR & TPR
#     os.makedirs("data/" + current_scheme + "/_lambda", exist_ok=True)
#     the_file = open("data/" + current_scheme + "/_lambda/FPR.pkl", "wb+")
#     pickle.dump(FPR_all_result, the_file)
#     the_file.close()
#     the_file = open("data/" + current_scheme + "/_lambda/TPR.pkl", "wb+")
#     pickle.dump(TPR_all_result, the_file)
#     the_file.close()


# def run_sumulation_group_varying_mu(current_scheme, DD_using, uncertain_scheme,
#                                     simulation_time):
#     varying_range = [6, 7, 8, 9, 10]
#
#     MTTSF_all_result = np.zeros(len(varying_range))
#     FPR_all_result = np.zeros(len(varying_range))
#     TPR_all_result = np.zeros(len(varying_range))
#
#     results = []
#     with concurrent.futures.ProcessPoolExecutor() as executor:
#         for vary_index in range(len(varying_range)):
#             particular_vul_result = []
#             for i in range(simulation_time):
#                 future = executor.submit(
#                     game_start, i, DD_using, uncertain_scheme, mu=varying_range[vary_index])  # scheme change here
#                 particular_vul_result.append(future)
#             results.append(particular_vul_result)
#
#         index = 0
#         for particular_vul_result in results:
#             total_time_for_all_sim = 0
#             for future in particular_vul_result:
#                 # change web server and database vul
#                 # MTTSF
#                 MTTSF_all_result[index] += future.result().lifetime
#                 # FPR & TPR
#                 FPR_all_result[index] += sum(
#                     future.result().FPR_history) / len(
#                     future.result().FPR_history)
#                 TPR_all_result[index] += sum(
#                     future.result().TPR_history) / len(
#                     future.result().TPR_history)
#                 total_time_for_all_sim += 1
#
#             FPR_all_result[
#                 index] = FPR_all_result[index] / total_time_for_all_sim
#             TPR_all_result[
#                 index] = TPR_all_result[index] / total_time_for_all_sim
#             MTTSF_all_result[index] = MTTSF_all_result[index] / simulation_time
#             index += 1
#
#     # SAVE to FILE (need to create directory manually)
#     # vary range
#     os.makedirs("data/" + current_scheme + "/mu", exist_ok=True)
#     the_file = open("data/" + current_scheme + "/mu/Range.pkl", "wb+")
#     pickle.dump(varying_range, the_file)
#     the_file.close()
#     # MTTSF
#     os.makedirs("data/" + current_scheme + "/mu", exist_ok=True)
#     the_file = open("data/" + current_scheme + "/mu/MTTSF.pkl", "wb+")
#     pickle.dump(MTTSF_all_result, the_file)
#     the_file.close()
#
#     # FPR & TPR
#     os.makedirs("data/" + current_scheme + "/mu", exist_ok=True)
#     the_file = open("data/" + current_scheme + "/mu/FPR.pkl", "wb+")
#     pickle.dump(FPR_all_result, the_file)
#     the_file.close()
#     the_file = open("data/" + current_scheme + "/mu/TPR.pkl", "wb+")
#     pickle.dump(TPR_all_result, the_file)
#     the_file.close()


def run_sumulation_group_varying_universal(current_scheme, DD_using, uncertain_scheme, simulation_time, variable_name,
                                           varying_range):
    MTTSF_all_result = np.zeros(len(varying_range))
    FPR_all_result = np.zeros(len(varying_range))
    TPR_all_result = np.zeros(len(varying_range))
    att_uncertainty_all_result = np.zeros(len(varying_range))
    def_uncertainty_all_result = np.zeros(len(varying_range))

    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for vary_index in range(len(varying_range)):
            particular_vul_result = []
            for i in range(simulation_time):
                future = eval(
                    "executor.submit(game_start, i, DD_using, uncertain_scheme, " + variable_name + "=varying_range[vary_index])")  # scheme change here
                particular_vul_result.append(future)
            results.append(particular_vul_result)

        index = 0
        for particular_vul_result in results:
            total_time_for_all_sim = 0
            for future in particular_vul_result:
                # change web server and database vul
                # MTTSF
                MTTSF_all_result[index] += future.result().lifetime
                # FPR & TPR
                FPR_all_result[index] += sum(
                    future.result().FPR_history) / len(
                    future.result().FPR_history)
                TPR_all_result[index] += sum(
                    future.result().TPR_history) / len(
                    future.result().TPR_history)
                # Uncertainty
                att_uncertainty_all_result[index] += sum(
                    future.result().att_uncertainty_history) / len(
                    future.result().att_uncertainty_history)
                def_uncertainty_all_result[index] += sum(
                    future.result().def_uncertainty_history) / len(
                    future.result().def_uncertainty_history)
                total_time_for_all_sim += 1

            FPR_all_result[
                index] = FPR_all_result[index] / total_time_for_all_sim
            TPR_all_result[
                index] = TPR_all_result[index] / total_time_for_all_sim
            MTTSF_all_result[index] = MTTSF_all_result[index] / simulation_time
            att_uncertainty_all_result[index] = att_uncertainty_all_result[
                                                    index] / total_time_for_all_sim
            def_uncertainty_all_result[index] = def_uncertainty_all_result[
                                                    index] / total_time_for_all_sim
            index += 1

    # SAVE to FILE (need to create directory manually)
    # vary range
    os.makedirs("data/" + current_scheme + "/" + variable_name, exist_ok=True)
    the_file = open("data/" + current_scheme + "/" + variable_name + "/Range.pkl", "wb+")
    pickle.dump(varying_range, the_file)
    the_file.close()
    # MTTSF
    the_file = open("data/" + current_scheme + "/" + variable_name + "/MTTSF.pkl", "wb+")
    pickle.dump(MTTSF_all_result, the_file)
    the_file.close()

    # FPR & TPR
    the_file = open("data/" + current_scheme + "/" + variable_name + "/FPR.pkl", "wb+")
    pickle.dump(FPR_all_result, the_file)
    the_file.close()
    the_file = open("data/" + current_scheme + "/" + variable_name + "/TPR.pkl", "wb+")
    pickle.dump(TPR_all_result, the_file)
    the_file.close()

    # uncertainty
    the_file = open("data/" + current_scheme + "/" + variable_name + "/defender_uncertainty.pkl", "wb+")
    pickle.dump(def_uncertainty_all_result, the_file)
    the_file.close()
    the_file = open("data/" + current_scheme + "/" + variable_name + "/attacker_uncertainty.pkl", "wb+")
    pickle.dump(att_uncertainty_all_result, the_file)
    the_file.close()
