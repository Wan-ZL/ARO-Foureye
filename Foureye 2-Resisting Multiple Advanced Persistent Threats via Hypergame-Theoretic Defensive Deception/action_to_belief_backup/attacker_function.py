#!/usr/bin/env python
# coding: utf-8

# In[1]:

from main import display
# from graph_function import *
from defender_function import *
import graph_function
import concurrent
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import copy
import time
import pickle


# In[2]:


# In[3]:


def att_strategy_option_matrix(CKC_number, strategy_number):
    strat_option = np.zeros((CKC_number, strategy_number))
    # R
    strat_option[0, 0] = 1
    if strategy_number-1==8: strat_option[0, 8] = 1
    # D
    strat_option[1, 0] = 1
    strat_option[1, 1] = 1
    if strategy_number-1==8: strat_option[1, 8] = 1
    # E
    strat_option[2, 0] = 1
    strat_option[2, 1] = 1
    strat_option[2, 2] = 1
    strat_option[2, 3] = 1
    strat_option[2, 4] = 1
    strat_option[2, 6] = 1
    if strategy_number-1==8: strat_option[2, 8] = 1
    # C2
    for i in range(strategy_number):
        strat_option[3, i] = 1
    strat_option[3, 7] = 0
    # M
    for i in range(strategy_number):
        strat_option[4, i] = 1
    strat_option[4, 7] = 0
    # DE
    for i in range(strategy_number):
        strat_option[5, i] = 1

    return strat_option


# In[4]:


def att_strategy_cost(strategy_number):
    #      preset cost
    attack_cost = np.zeros(strategy_number)
    attack_cost[0] = 1
    attack_cost[1] = 3  # test: orignial 3
    attack_cost[2] = 3  # test: orignial 3
    attack_cost[3] = 3  # test: orignial 3
    attack_cost[4] = 1
    attack_cost[5] = 3  # test: orignial 3
    attack_cost[6] = 2  # test: orignial 2
    attack_cost[7] = 3  # test: orignial 3
    if strategy_number-1==8: attack_cost[8] = 0

    return attack_cost


# In[5]:


def update_strategy_probability(opponent_strat_history):
    return_result = np.zeros((len(opponent_strat_history), len(opponent_strat_history[0])))
    sum_botton = np.sum(opponent_strat_history, axis=1)
    for k in range(len(opponent_strat_history)):
        for j in range(len(opponent_strat_history[0])):
            if sum_botton[k] == 0:
                return_result[k][j] = 1 / len(opponent_strat_history[0])
            else:
                return_result[k][j] = opponent_strat_history[k][j] / sum_botton[k]

    return return_result


# In[6]:


def attacker_uncertainty_update(att_in_system_time, att_detect, dec, uncertain_scheme, decision_scheme):
    lambd = 1  # was 2

    # df = 1 + (1 - att_detect) * dec
    uncertainty = 1 - math.exp((-lambd) * (1 + (1 - att_detect) * dec) / att_in_system_time)

    # (scheme change here!)
    if decision_scheme == 0:
        return 1
    else:
        if uncertain_scheme:
            return uncertainty # for test. orignial: uncertainty
        else:
            return 0            # for test. orignial: 0


# In[7]:


# APV: value of an intermediate node i in an attack path, APV=-1 means the node is detected as Honeypot
def calc_APV(G_att, G_real, node_ID, attack_cost, attack_detect_prob):
    if G_att.nodes[node_ID]["compromised_status"]:
        return 1

    att_detect_honey = False
    if G_real.nodes[node_ID]["honeypot"] == 1:
        if random.random() < attack_detect_prob:
            att_detect_honey = True
    elif G_real.nodes[node_ID]["honeypot"] == 2:
        if random.random() < (attack_detect_prob / 2):
            att_detect_honey = True
    # elif G_real.nodes[node_ID]["evicted_mark"]:
    #     if random.random() < attack_detect_prob:
    #         att_detect_evicted = True

    if att_detect_honey:
        # if the node is honeypot, the APV for honeypot is 0.
        return -1
    else:
        return (1 - (attack_cost / 3)) * G_att.nodes[node_ID]["normalized_vulnerability"]


# Attack Impact of given attack k of attacker i
# new_compromised_list is new compromised node IDs (do not include already compromised node)
# Return: ai_{ik}
def attack_impact(G, new_compromised_list, node_number):
    if len(new_compromised_list) == 0:
        return 0

    N = G.number_of_nodes()

    total_criticality = 0
    for n in new_compromised_list:
        total_criticality += G.nodes[n]["criticality"]
    ai = total_criticality / node_number #G.number_of_nodes() #/ G.number_of_nodes()  # for test, original uncomment
    return ai


# In[9]:


def get_attacker_network_list(attacker_list):
    G_att_list = []
    for attacker in attacker_list:
        G_att_list.append(attacker.network)
    return G_att_list


# In[ ]:


# In[ ]:


# In[10]:


# AS1 – Monitoring attack
# (keep trying until get one)
# return: a dictionary contain all information
def attack_AS_1(G_real, G_att, G_def, node_info_list, monit_time, attack_detect_prob):
    # probability to share information
    attack_cost = 1
    attack_result = {"attack_cost": attack_cost, "ids": []}

    node_id_set = list(G_att.nodes())

    not_get_one = True
    while not_get_one:
        random_id = random.choice(node_id_set)
        # Check if detect Honeypot
        if calc_APV(G_att, G_real, random_id, attack_cost, attack_detect_prob) != -1:
            not_get_one = False
            # failed for evicted node
            if not G_real.nodes[random_id]["evicted_mark"]:
                if random.random() <= G_real.nodes[random_id]["normalized_vulnerability"] * math.exp(
                        -1 / monit_time):  # success rate is based on real graph
                    node_info_list.append(G_att.nodes[random_id]["id"])
                    attack_result["ids"].append(random_id)

    return attack_result


# In[11]:


# AS2 – Social engineering
# input: node_info_list: nodes with information collected (dictionary type)
#
# return: a dictionary with "attack cost", and compromised "ids". "ids" is empty if unsuccessful
def attack_AS_2(node_info_list, G_real, G_att, G_def, P_fake,
                attack_detect_prob, location, compromise_probability):
    attack_cost = 3
    attack_result = {"attack_cost": attack_cost, "ids": []}
    max_number_of_phishing = 5 # was 10

    # fake key probability
    if random.random() <= P_fake:
        if display:
            print("get fake key, failed to compromise")
        return attack_result

    # if outside
    if location is None:
        if node_info_list:
            target_node_id = random.choice(node_info_list)
            if len(node_info_list) > max_number_of_phishing:
                target_node_list = random.sample(node_info_list, max_number_of_phishing)
            else:
                target_node_list = node_info_list

            for target_node in target_node_list:
                # fail if node doesn't exist
                if not G_real.has_node(target_node):
                    continue
                # failed for evicted node
                if G_real.nodes[target_node]["evicted_mark"]:
                    continue
                # compromise
                compromise_probability.append(G_real.nodes[target_node]["normalized_vulnerability"])
                if random.random() < G_real.nodes[target_node]["normalized_vulnerability"]:
                    attack_result["ids"].append(target_node)
                    G_real.nodes[target_node]["compromised_status"] = True
                    G_att.nodes[target_node]["compromised_status"] = True
        return attack_result

    # if inside
    max_APV_id = None
    max_APV = 0
    att_neighbors = [n for n in G_att[location]]

    # decide which node to compromise
    for node_id in att_neighbors:
        if G_real.has_node(node_id):
            if calc_APV(G_att, G_real, G_att.nodes[node_id]["id"], attack_cost,
                        attack_detect_prob) >= max_APV:  # choose node with APV
                max_APV = calc_APV(G_att, G_real, G_att.nodes[node_id]["id"],
                                   attack_cost, attack_detect_prob)
                max_APV_id = G_att.nodes[node_id]["id"]

    if max_APV_id is None:
        if display:
            print("no legitimate node in collection list \U0001F630")
        return attack_result

    # failed for evicted node
    if G_real.nodes[max_APV_id]["evicted_mark"]:
        return attack_result

    # collusive attack
    if G_att.nodes[max_APV_id]["compromised_status"] and G_real.nodes[
        max_APV_id]["compromised_status"]:
        attack_result["ids"].append(max_APV_id)
        if display: print("AS2: collusive attack")
        return attack_result

    # compromise attempt
    if random.random() < G_real.nodes[max_APV_id]["normalized_vulnerability"]:
        attack_result["ids"].append(max_APV_id)
        # set it compromised for real & att
        G_real.nodes[max_APV_id]["compromised_status"] = True
        G_att.nodes[max_APV_id]["compromised_status"] = True
    else:
        if display:
            print("AS2: unsuccessful on", max_APV_id, "with vul",
                  G_real.nodes[max_APV_id]["normalized_vulnerability"])
    return attack_result


# In[12]:


# AS3 – Botnet-based attack
# (a legitimate node with more than one compromised node will be tried more than one times)
# return: attack_result["ids"] is new compromised ids
def attack_AS_3(collection_list, G_real, G_att, G_def, P_fake,
                attack_detect_prob):
    attack_result = {"attack_cost": 3, "ids": []}

    if random.random() <= P_fake:
        if display:
            print("get fake key, failed to compromise")
        return attack_result

    attacked_adjacent = []

    for node_id in collection_list:
    # for node_id in G_real.nodes():      # test, original: above one
        if not G_real.nodes[node_id]["evicted_mark"]:
            # if attacker detect deception, use real network
            if random.random() < attack_detect_prob:
                attacked_adjacent += graph_function.adjacent_node(G_real, node_id)
            else:
                attacked_adjacent += graph_function.adjacent_node(G_att, node_id)

    attacked_adjacent = list(set(attacked_adjacent))
    # print(attacked_adjacent)
    for n in attacked_adjacent:
        if G_att.nodes[n]["compromised_status"] and G_real.nodes[n][
            "compromised_status"]:
            # collusive attack
            attack_result["ids"].append(n)
        else:
            if random.random() <= G_real.nodes[n]["normalized_vulnerability"]:
                G_real.nodes[n]["compromised_status"] = True
                G_att.nodes[n]["compromised_status"] = True

                attack_result["ids"].append(n)
    # print("attack_result")
    # print(attack_result["ids"])
    return attack_result


# In[13]:


# AS4 – Distributed Denial-of-Service (DDoS)
# return: attack_result["ids"] is the node that Unknow Vulnerability(UV) increased


def attack_AS_4(G_real, G_att, G_def, attack_detect_prob, P_fake, attacker_locaton):
    attack_cost = 3
    attack_result = {"attack_cost": attack_cost, "ids": []}

    if random.random() < attack_detect_prob:
        attacker_adjacent = graph_function.adjacent_node(G_real, attacker_locaton)
    else:
        attacker_adjacent = graph_function.adjacent_node(G_att, attacker_locaton)

    max_APV_id = None
    max_APV = 0

    # perform DDoS
    for node_id in attacker_adjacent:
        if G_real.has_node(node_id):
            G_real.nodes[node_id]["unknown vulnerability"][0] = min(
                G_real.nodes[node_id]["unknown vulnerability"][0] * 1.1, 10)
            G_att.nodes[node_id]["unknown vulnerability"][0] = min(
                G_att.nodes[node_id]["unknown vulnerability"][0] * 1.1, 10)
            G_def.nodes[node_id]["unknown vulnerability"][0] = min(
                G_def.nodes[node_id]["unknown vulnerability"][0] * 1.1, 10)

    # update nodes attribute
    graph_function.update_vul(G_real)
    graph_function.update_vul(G_att)
    graph_function.update_vul(G_def)

    if random.random() <= P_fake:
        if display:
            print("get fake key, failed to compromise")
        return attack_result

    # decide which node to compromise
    for node_id in attacker_adjacent:
        if G_att.has_node(node_id):
            if calc_APV(G_att, G_real, G_att.nodes[node_id]["id"], attack_cost,
                        attack_detect_prob) >= max_APV:  # choose node with APV
                max_APV = calc_APV(G_att, G_real, G_att.nodes[node_id]["id"],
                                   attack_cost, attack_detect_prob)
                max_APV_id = G_att.nodes[node_id]["id"]

    if max_APV_id is None:
        if display:
            print("attacker have no legitimate adjacent node \U0001F630")
        return attack_result

    # failed for evicted node
    if G_real.nodes[max_APV_id]["evicted_mark"]:
        return attack_result

    # collusive attack
    if G_att.nodes[max_APV_id]["compromised_status"] and G_real.nodes[
        max_APV_id]["compromised_status"]:
        attack_result["ids"].append(max_APV_id)
        if display: print("AS4: collusive attack")
        return attack_result

    # compromise attempt
    if random.random() < G_real.nodes[max_APV_id]["normalized_vulnerability"]:
        attack_result["ids"].append(max_APV_id)
        # set it compromised
        G_real.nodes[max_APV_id]["compromised_status"] = True
        G_att.nodes[max_APV_id]["compromised_status"] = True

    else:
        if display:
            print("AS4: unsuccessful on", max_APV_id, "with vul",
                  G_real.nodes[max_APV_id]["normalized_vulnerability"])

    return attack_result


# In[14]:


# AS5 – Zero-day attacks
def attack_AS_5(G_real, G_att, G_def, attacker_locaton, attack_detect_prob):
    attack_cost = 1

    if random.random() < attack_detect_prob:
        attacked_adjacent = graph_function.adjacent_node(G_real, attacker_locaton)
    else:
        attacked_adjacent = graph_function.adjacent_node(G_att, attacker_locaton)

    attack_result = {"attack_cost": attack_cost, "ids": []}

    max_APV_id = None
    max_APV = 0
    # decide which node to compromise
    for n in attacked_adjacent:
        if calc_APV(G_att, G_real, n, attack_cost, attack_detect_prob) >= max_APV:
            max_APV_id = n
            max_APV = calc_APV(G_att, G_real, n, attack_cost, attack_detect_prob)

    if max_APV_id is None:
        if display:
            print("no legitimate neighbor node \U0001F630")
        return attack_result

    # failed for evicted node
    if G_real.nodes[max_APV_id]["evicted_mark"]:
        return attack_result

    # collusive attack
    if G_att.nodes[max_APV_id]["compromised_status"] and G_real.nodes[
        max_APV_id]["compromised_status"]:
        attack_result["ids"].append(max_APV_id)
        if display: print("AS5: collusive attack")
        return attack_result

    #     try compromising
    if random.uniform(0, 10) <= G_real.nodes[max_APV_id]["normalized_vulnerability"]:
        G_real.nodes[max_APV_id]["compromised_status"] = True
        G_att.nodes[max_APV_id]["compromised_status"] = True

        # attack_result["ids"].append(max_APV_id)

    return attack_result


# In[15]:


# AS6 – Breaking encryption


def attack_AS_6(G_real, G_att, G_def, attacker_locaton, P_fake,
                attack_detect_prob):
    attack_cost = 3
    attack_result = {"attack_cost": attack_cost, "ids": []}

    if random.random() <= P_fake:
        if display:
            print("get fake key, failed to compromise")
        return attack_result

    if random.random() < attack_detect_prob:
        attacked_adjacent = graph_function.adjacent_node(G_real, attacker_locaton)
    else:
        attacked_adjacent = graph_function.adjacent_node(G_att, attacker_locaton)

    # decide which node to compromise
    max_APV_id = None
    max_APV = 0
    for n in attacked_adjacent:
        if calc_APV(G_att, G_real, n, attack_cost,
                    attack_detect_prob) >= max_APV:
            max_APV_id = n
            max_APV = calc_APV(G_att, G_real, n, attack_cost,
                               attack_detect_prob)

    if max_APV_id is None:
        if display:
            print("no legitimate neighbor node \U0001F630")
        return attack_result

    # failed for evicted node
    if G_real.nodes[max_APV_id]["evicted_mark"]:
        return attack_result

    # collusive attack
    if G_att.nodes[max_APV_id]["compromised_status"] and G_real.nodes[
        max_APV_id]["compromised_status"]:
        attack_result["ids"].append(max_APV_id)
        if display: print("AS6: collusive attack")
        return attack_result

    # compromise attempt
    if random.random() <= sum(
            G_real.nodes[max_APV_id]["encryption vulnerability"]) / len(
        G_real.nodes[max_APV_id]["encryption vulnerability"]):
        G_real.nodes[max_APV_id]["compromised_status"] = True
        G_att.nodes[max_APV_id]["compromised_status"] = True

        attack_result["ids"].append(max_APV_id)
    else:
        if display:
            print("AS6: unsuccessful on", max_APV_id, "with APV", max_APV)

    return attack_result


# In[16]:


# AS7 – Fake identity
def attack_AS_7(G_real, G_att, G_def, attacker_locaton, P_fake,
                attack_detect_prob):
    attack_cost = 2
    attack_result = {"attack_cost": attack_cost, "ids": []}

    # Increase EV
    if random.random() < attack_detect_prob:
        attacker_adjacent = graph_function.adjacent_node(G_real, attacker_locaton)
    else:
        attacker_adjacent = graph_function.adjacent_node(G_att, attacker_locaton)

    for node_id in attacker_adjacent:
        if G_real.has_node(node_id):
            length = len(G_real.nodes[node_id]["encryption vulnerability"])
            for index in range(length):
                G_real.nodes[node_id]["encryption vulnerability"][index] = min(
                    G_real.nodes[node_id]["encryption vulnerability"][index] *
                    1.1, 10)
                G_att.nodes[node_id]["encryption vulnerability"][index] = min(
                    G_att.nodes[node_id]["encryption vulnerability"][index] *
                    1.1, 10)
                G_def.nodes[node_id]["encryption vulnerability"][index] = min(
                    G_def.nodes[node_id]["encryption vulnerability"][index] *
                    1.1, 10)

    # update nodes attribute
    graph_function.update_vul(G_real)
    graph_function.update_vul(G_att)
    graph_function.update_vul(G_def)

    if random.random() <= P_fake:
        if display:
            print("get fake key, failed to compromise")
        return attack_result

    # decide which node to compromise
    max_APV_id = None
    max_APV = 0
    for n in attacker_adjacent:
        if calc_APV(G_att, G_real, n, attack_cost,
                    attack_detect_prob) >= max_APV:
            max_APV_id = n
            max_APV = calc_APV(G_att, G_real, n, attack_cost,
                               attack_detect_prob)

    if max_APV_id is None:
        if display:
            print("no legitimate neighbor node \U0001F630")
        return attack_result

    # failed for evicted node
    if G_real.nodes[max_APV_id]["evicted_mark"]:
        return attack_result

    # collusive attack
    if G_att.nodes[max_APV_id]["compromised_status"] and G_real.nodes[
        max_APV_id]["compromised_status"]:
        attack_result["ids"].append(max_APV_id)
        if display: print("AS6: collusive attack")
        return attack_result

    # try compromising
    for index in range(len(G_att.nodes[n]["encryption vulnerability"])):
        if random.uniform(
                0, 10
        ) <= G_real.nodes[max_APV_id]["encryption vulnerability"][index]:
            G_real.nodes[max_APV_id]["compromised_status"] = True
            G_att.nodes[max_APV_id]["compromised_status"] = True

            attack_result["ids"].append(max_APV_id)
            break

    return attack_result


# In[17]:


# AS8 – Data exfiltration
def attack_AS_8(G_real, G_att, G_def, compromised_nodes, attacker_locaton,
                P_fake, attack_detect_prob, node_size_multiplier):
    if random.random() < attack_detect_prob:
        attacked_adjacent = graph_function.adjacent_node(G_real, attacker_locaton)
    else:
        attacked_adjacent = graph_function.adjacent_node(G_att, attacker_locaton)

    attack_cost = 3
    attack_result = {
        "attack_cost": attack_cost,
        "ids": [],
        "data_exfiltrated": False
    }

    # decide which node to compromise
    max_APV_id = None
    max_APV = 0
    for n in attacked_adjacent:
        if calc_APV(G_att, G_real, n, attack_cost,
                    attack_detect_prob) >= max_APV:
            max_APV_id = n
            max_APV = calc_APV(G_att, G_real, n, attack_cost,
                               attack_detect_prob)

    if max_APV_id is None:
        if display:
            print("no legitimate neighbor node \U0001F630")
        return attack_result

    # failed for evicted node
    if G_real.nodes[max_APV_id]["evicted_mark"]:
        return attack_result

    if G_att.nodes[max_APV_id]["compromised_status"] and G_real.nodes[
        max_APV_id]["compromised_status"]:
        # collusive attack
        attack_result["ids"].append(max_APV_id)
    else:
        if random.random() <= G_real.nodes[max_APV_id]["normalized_vulnerability"]:
            # compromise attempt
            G_real.nodes[max_APV_id]["compromised_status"] = True
            G_att.nodes[max_APV_id]["compromised_status"] = True

            attack_result["ids"].append(max_APV_id)

    if random.random() <= P_fake:
        if display:
            print("get fake key, failed to compromise")
    else:
        compromised_nodes.append(max_APV_id)

    # data exfiltration
    Thres_c = 30 * node_size_multiplier  # 30  # pre-set value
    total_compromised_importance = 0
    for node_id in compromised_nodes:
        if G_real.has_node(node_id):
            if G_real.nodes[node_id]["compromised_status"]:
                total_compromised_importance += G_real.nodes[node_id][
                    "importance"]

    if total_compromised_importance > Thres_c:
        if display: print("Data exfiltration success")
        if display:
            print("total collected importance is",
                  total_compromised_importance)
        attack_result["data_exfiltrated"] = True
    else:
        if display: print("Data exfiltration failed")
        attack_result["data_exfiltrated"] = False

    return attack_result


def attack_AS_9(att_location, G_def):
    G_def.nodes[att_location]["stealthy_status"] = True

    attack_result = {"attack_cost": 0, "ids": []}
    return attack_result


def get_network_list(attacker_list):
    G_att_list = []
    for attacker in attacker_list:
        G_att_list.append(attacker.network)

    return G_att_list


# In[20]:


def get_average_detect_prob(attacker_list):
    average_value = 0
    if not attacker_list:  # if no attacker in system
        return average_value

    for attacker in attacker_list:
        average_value += attacker.detect_prob
    average_value = average_value / len(attacker_list)

    return average_value


# In[ ]:


def get_detect_prob_list(attacker_list):
    detect_prob_list = []
    for attacker in attacker_list:
        detect_prob_list.append(attacker.detect_prob)
    return detect_prob_list


# In[ ]:


def get_P_fake_list(attacker_list):
    P_fake_list = []
    for attacker in attacker_list:
        P_fake_list.append(attacker.P_fake)
    return P_fake_list


# In[ ]:


def get_CKC_list(attacker_list):
    CKC_list = []
    for attacker in attacker_list:
        CKC_list.append(attacker.CKC_position)

    return CKC_list


def get_strategy_list(attacker_list):
    stra_list = []
    for attacker in attacker_list:
        stra_list.append(attacker.chosen_strategy)
    return stra_list


def get_location_list(attacker_list):
    location_list = []
    for attacker in attacker_list:
        location_list.append(attacker.location)
    return location_list


def get_ID_list(attacker_list):
    ID_list = []
    for attacker in attacker_list:
        ID_list.append(attacker.attacker_ID)
    return ID_list


# In[3]:


# Averaged Impact ai_k
def get_averaged_impact(attacker_list, attacker_template):
    averaged_impact = np.zeros(attacker_template.strategy_number)
    if not attacker_list:
        return averaged_impact

    for attacker in attacker_list:
        averaged_impact += attacker.impact_record
    averaged_impact = averaged_impact / len(attacker_list)

    return averaged_impact


# Attack Impact at time 't'
def get_overall_attacker_impact_per_game(attacker_list, attacker_template):
    if len(attacker_list) == 0:
        return 0

    ai_per_strategy = np.zeros(attacker_template.strategy_number)
    ai_per_strategy_counter = np.zeros(attacker_template.strategy_number)

    for attacker in attacker_list:
        att_chosen_strategy = attacker.chosen_strategy
        att_impact = attacker.impact_record[att_chosen_strategy]
        ai_per_strategy[att_chosen_strategy] += att_impact
        ai_per_strategy_counter[att_chosen_strategy] += 1

    ai_k = [ai_per_strategy[index] / ai_per_strategy_counter[index]
            if ai_per_strategy_counter[index] != 0 else 0
            for index in range(len(ai_per_strategy))]

    overall_ai = sum(ai_k) / len(attacker_list)
    return overall_ai



def _2darray_normalization(_2d_array):
    sum_array = np.ones(_2d_array.shape)/_2d_array.shape[1]
    for index in range(_2d_array.shape[0]):
        if np.sum(_2d_array[index]) == 0:
            continue
        else:
            sum_array[index] = _2d_array[index]/np.sum(_2d_array[index])
    return sum_array

def array_normalization(array):
    if np.sum(array) == 0:
        return np.ones(array.shape)/len(array)
    else:
        return array/np.sum(array)


def attacker_class_choose_strategy(self, def_strategy_number,
                                   defend_cost_record, defend_impact_record):
    if random.random() > self.uncertainty:
        # certain level
        S_j = np.ones(self.strategy_number) / self.strategy_number

        c_kj = _2darray_normalization(self.observed_strategy_count)
        p_k = array_normalization(self.observed_CKC_count)
        for j in range(len(S_j)):
            temp_S = 0
            for k in range(len(p_k)):
                temp_S += p_k[k] * c_kj[k][j]
            S_j[j] = temp_S
        self.S_j = S_j

        # eq. 17
        utility = np.zeros((self.strategy_number, def_strategy_number))
        for i in range(self.strategy_number):
            for j in range(def_strategy_number):
                utility[i,
                        j] = (self.impact_record[i] +
                              defend_cost_record[j] / 3) - (
                                     self.strat_cost[i] / 3 + defend_impact_record[j])

        # normalization range
        a = 1
        b = 9

        # eq. 8
        EU_C = np.zeros(self.strategy_number)
        for i in range(self.strategy_number):
            for j in range(def_strategy_number):
                EU_C[i] += S_j[j] * utility[i, j]


        # eq. 9
        EU_CMS = np.zeros(self.strategy_number)
        for i in range(self.strategy_number):
            w = np.argmin(utility[i])  # min utility index
            EU_CMS[i] = self.strategy_number * S_j[w] * utility[i][w]

        HEU = EU_C

        # Min-Max Normalization
        if (max(HEU) - min(HEU)) != 0:
            HEU = a + (HEU - min(HEU)) * (b - a) / (max(HEU) - min(HEU))
        else:
            HEU = np.ones(self.strategy_number) * a
        self.HEU = HEU  # uncertainty case doesn't consider as real HEU

        # eq. 23
        AHEU = np.zeros(self.strategy_number)
        for index in range(self.strategy_number):
            AHEU[index] = HEU[index] * self.strat_option[
                self.CKC_position, index]  # for Table 4
    else:
        # uncertain level
        # Whole Random:
        AHEU = np.ones(self.strategy_number)
        # Limited Random:
        # if 0 <= self.CKC_position <= 1: # outside type
        #     AHEU = np.zeros(self.strategy_number)
        #     AHEU[0] = 1
        #     AHEU[1] = 1
        # else:                           # inside type
        #     AHEU = np.ones(self.strategy_number)

    # Selection Scheme
    if self.decision_scheme == 0:  # for random selection scheme
        self.chosen_strategy = random.choices(range(self.strategy_number))[0]
    elif self.decision_scheme == 1 or self.decision_scheme == 2:  # HEU-based selection scheme
        if sum(AHEU) == 0:  # fix python 3.5 error
            self.chosen_strategy = random.choices(range(self.strategy_number))[0]
        else:
            self.chosen_strategy = random.choices(range(self.strategy_number), weights=AHEU, k=1)[0]
    else:
        raise Exception("Error: Unknown decision_scheme")

    # if sum(AHEU) == 0:  # fix python 3.5 error
    #     self.chosen_strategy = random.choices(range(self.strategy_number))[0]
    # else:
    #     self.chosen_strategy = random.choices(range(self.strategy_number), weights=AHEU, k=1)[0]

    return self.chosen_strategy


# In[ ]:


def attacker_class_execute_strategy(self, G_real, G_def, node_size_multiplier, compromise_probability):
    return_value = False
    attack_result = {"attack_cost": 0, "ids": []}

    if self.chosen_strategy == 0:

        attack_result = attack_AS_1(G_real, self.network, G_def,
                                    self.collection_list, self.monit_time, self.detect_prob)
        if attack_result["ids"]:
            return_value = True
            attack_result["ids"] = []  # reset to empty for avoiding attacker moves to observed location.

    elif self.chosen_strategy == 1:
        attack_result = attack_AS_2(self.collection_list, G_real, self.network, G_def,
                                    self.P_fake[0], self.detect_prob, self.location, compromise_probability)
        if attack_result["ids"]:
            self.compromised_nodes.extend(attack_result["ids"])

            #  decrease collection list
            if self.location is None:
                for node_id in attack_result["ids"]:
                    self.collection_list.remove(node_id)

            return_value = True
        else:
            if display:
                print("attack 2 failed")

    elif self.chosen_strategy == 2:
        attack_result = attack_AS_3(self.collection_list, G_real, self.network, G_def, self.P_fake[0],
                                    self.detect_prob)
        if attack_result["ids"]:
            self.compromised_nodes.extend(attack_result["ids"])
            return_value = True
        else:
            if display:
                print("attack 3 failed")

    elif self.chosen_strategy == 3:
        attack_result = attack_AS_4(G_real, self.network, G_def,
                                    self.detect_prob, self.P_fake[0], self.location)
        if attack_result["ids"]:
            self.compromised_nodes.extend(attack_result["ids"])
            return_value = True
        else:
            if display:
                print("attack 4 failed")

    elif self.chosen_strategy == 4:
        attack_result = attack_AS_5(G_real, self.network, G_def, self.location,
                                    self.detect_prob)
        if attack_result["ids"]:
            self.compromised_nodes.extend(attack_result["ids"])
            return_value = True
        else:
            if display:
                print("attack 5 failed")

    elif self.chosen_strategy == 5:
        attack_result = attack_AS_6(G_real, self.network, G_def, self.location,
                                    self.P_fake[0], self.detect_prob)
        if attack_result["ids"]:
            self.compromised_nodes.extend(attack_result["ids"])
            return_value = True
        else:
            if display:
                print("attack 6 failed")

    elif self.chosen_strategy == 6:
        attack_result = attack_AS_7(G_real, self.network, G_def, self.location,
                                    self.P_fake[0], self.detect_prob)
        if attack_result["ids"]:
            self.compromised_nodes.extend(attack_result["ids"])
            return_value = True
        else:
            if display:
                print("attack 7 failed")

    elif self.chosen_strategy == 7:
        attack_result = attack_AS_8(G_real, self.network, G_def,
                                    self.compromised_nodes, self.location,
                                    self.P_fake[0], self.detect_prob, node_size_multiplier)
        if attack_result["ids"]:
            self.compromised_nodes.extend(attack_result["ids"])

        if attack_result["data_exfiltrated"]:
            self.exfiltrate_data = True
            return_value = True
        else:
            if display:
                print("attack 8 failed")

    else:
        if self.location is not None:
            attack_result = attack_AS_9(self.location, G_def)
            if display:
                print("attack 9 executed")

    self.impact_record[self.chosen_strategy] = attack_impact(G_real, attack_result["ids"], self.node_number)    # original: uncomment
    # self.impact_count[self.chosen_strategy] += attack_impact(G_real, attack_result["ids"])   # for test
    # self.strategy_count[self.chosen_strategy] += 1
    # # temp_list = np.zeros(self.strategy_number)
    # for index in range(len(self.impact_record)):
    #     if not self.strategy_count[index] == 0:
    #         self.impact_record[index] = self.impact_count[index] / self.strategy_count[index]




    # print("Impact is:")
    # print(self.impact_record[self.chosen_strategy])
    if attack_result["ids"]:
        self.location = random.choice(attack_result["ids"])

    return return_value


# In[ ]:


class attacker_class:
    def __init__(self, game, attacker_ID):
        if display:
            print("create attacker")
        self.attacker_ID = attacker_ID
        self.network = copy.deepcopy(game.graph.network)  # attacker's view
        self.node_number = game.graph.node_number
        self.strategy_number = game.strategy_number
        self.collusion_attack_probability = game.collusion_attack_probability
        self.collection_list = []
        self.location = None
        self.impact_count = np.zeros(self.strategy_number)   # for test
        self.strategy_count = np.zeros(self.strategy_number)   # for test
        self.impact_record = np.ones(self.strategy_number)  # attacker believe all strategy have full impact initially
        self.strat_cost = att_strategy_cost(self.strategy_number)
        self.strat_option = att_strategy_option_matrix(game.CKC_number, self.strategy_number)  # Table 4
        self.belief_context = [1 / (game.CKC_number + 1)] * (game.CKC_number + 1)
        self.CKC_position = 0
        self.CKC_number = game.CKC_number
        self.prob_believe_opponent = np.zeros((game.CKC_number + 1,self.strategy_number))
        self.in_system_time = 1
        self.P_fake = [0]  # (Make variable mutable) fake key condition controlled by DS7
        self.monit_time = 1
        self.detect_prob = random.uniform(0, 0.5)
        self.decision_scheme = game.decision_scheme
        self.chosen_strategy = 0
        self.in_honeynet = False
        self.uncertain_scheme = game.uncertain_scheme_att
        print(f"game.decision_scheme {game.decision_scheme}")
        if game.decision_scheme == 0:
            self.uncertainty = 1
        else:
            if self.uncertain_scheme:
                # 1  # 100% uncertainty at beginning (scheme change here!)
                self.uncertainty = 1 # test, orignial 1
            else:
                self.uncertainty = 0    # test, orignial 0
        self.HEU = np.zeros(self.strategy_number)
        self.compromised_nodes = []
        self.EU_C = None
        self.EU_CMS = None
        self.exfiltrate_data = False
        # self.observed_strategy_count = np.zeros(self.strategy_number) # old
        self.observed_strategy_count = np.zeros((self.CKC_number, self.strategy_number))
        self.observed_CKC_count = np.zeros(self.CKC_number)
        self.S_j = np.ones(self.strategy_number) / self.strategy_number

    choose_strategy = attacker_class_choose_strategy

    execute_strategy = attacker_class_execute_strategy

    def reset_in_system_time(self):
        self.in_system_time = 1
        return self.in_system_time

    def next_stage(self):
        if self.CKC_position != 5:
            self.CKC_position += 1

        # test reset impact
        # self.impact_count = np.zeros(self.strategy_number)  # for test
        # self.strategy_count = np.zeros(self.strategy_number)  # for test


    def reset_attribute(self):
        pass

    def observe_opponent(self, chosen_strategy_list):
        # observe opponent action in one game

        # observe CKC
        observed_CKC_id = self.CKC_position

        observed_action_list = []
        for the_strategy in chosen_strategy_list:
            if random.random() > self.uncertainty:
                observed_action_list.append(the_strategy)

        if len(observed_action_list) == 0:
            observed_action_list.append(random.randrange(0, len(self.observed_strategy_count)))

        for observed_action_id in observed_action_list:
            self.observed_strategy_count[observed_CKC_id, observed_action_id] += 1    # for testt
            self.observed_CKC_count[observed_CKC_id] += 1                         # for test

    def observe_opponent_old(self, chosen_strategy_list):
        # observe opponent action in one game
        temp_observed_strategy_list = np.zeros(self.observed_strategy_count.shape)
        for the_strategy in chosen_strategy_list:
            # Observe column player's strategy
            if random.random() > self.uncertainty:
                temp_observed_strategy_list[the_strategy] += 1

        # if not observe anything, randomly guess one.
        if sum(temp_observed_strategy_list) == 0:
            temp_observed_strategy_list[random.randrange(0,len(temp_observed_strategy_list))] += 1
        # count update
        self.observed_strategy_count = self.observed_strategy_count + temp_observed_strategy_list


    def update_attribute(self, dec):
        # monitor time
        self.monit_time += 1

        # if in_system
        if self.CKC_position >= 2:
            self.in_system_time += 1

        # belief context
        self.belief_context[0] = 1 - sum(self.belief_context[1:])

        # in honeypot
        if self.location is not None:
            if self.network.nodes[self.location]["type"] == 3:
                self.in_honeynet = True
            else:
                self.in_honeynet = False

        # uncertainty
        self.uncertainty = attacker_uncertainty_update(self.in_system_time,
                                                       self.detect_prob, dec,
                                                       self.uncertain_scheme, self.decision_scheme)

    def random_moving(self):
        if self.location is None:
            return

        neighbor_list = [i for i in self.network[self.location]]
        compromised_neighbor_list = [self.location]  # allow attacker stands still
        for index in neighbor_list:
            if self.network.nodes[index]["compromised_status"]:
                compromised_neighbor_list.append(index)

        self.location = random.choice(compromised_neighbor_list)


# In[ ]:


# In[ ]:
