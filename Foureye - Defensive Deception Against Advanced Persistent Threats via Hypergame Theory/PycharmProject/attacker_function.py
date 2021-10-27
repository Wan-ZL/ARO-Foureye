#!/usr/bin/env python
# coding: utf-8

# In[34]:


import random
import numpy as np
from main import display
from defender_function import *
import numpy as np
import random
import math
import copy
import graph_function


# In[ ]:





# In[35]:


def att_strategy_option_matrix(CKC_number, strategy_number):
        strat_option = np.zeros((CKC_number, strategy_number))
        # R
        strat_option[0, 0] = 1
        # D
        strat_option[1, 0] = 1
        strat_option[1, 1] = 1
        # E
        strat_option[2, 1] = 1
        strat_option[2, 2] = 1
        strat_option[2, 3] = 1
        strat_option[2, 4] = 1
        strat_option[2, 6] = 1
        # C2
        for i in range(strategy_number - 1):
            strat_option[3, i] = 1
        # M
        for i in range(strategy_number - 1):
            strat_option[4, i] = 1
        # DE
        for i in range(strategy_number):
            strat_option[5, i] = 1
        
        return strat_option


# In[36]:


def att_strategy_cost(strategy_number):
#      preset cost
    attack_cost = np.zeros(strategy_number)
    attack_cost[0] = 1
    attack_cost[1] = 3
    attack_cost[2] = 3
    attack_cost[3] = 3
    attack_cost[4] = 1
    attack_cost[5] = 3
    attack_cost[6] = 2
    attack_cost[7] = 3
    
    return attack_cost


# In[37]:


def update_strategy_probability(opponent_strat_history):
    return_result = np.zeros((len(opponent_strat_history), len(opponent_strat_history[0])))
    sum_botton = np.sum(opponent_strat_history, axis=1)
    for k in range(len(opponent_strat_history)):
        for j in range(len(opponent_strat_history[0])):
            if sum_botton[k] == 0:
                return_result[k][j] = 1/len(opponent_strat_history[0])
            else:
                return_result[k][j] = opponent_strat_history[k][j]/sum_botton[k]
    
    return return_result
    
    


# In[38]:


def attacker_uncertainty_update(att_in_system_time, att_detect, dec, uncertain_scheme, _lambda):
    # _lambda = 0.8 # was 2

    df = 1 + (1-att_detect) * dec
    uncertainty = 1 - math.exp((-_lambda) * (df)/att_in_system_time)
    
# (scheme change here!) 
    if uncertain_scheme:
        return uncertainty
    else:
        return 0


# In[39]:


# APV: value of an intermediate node i in an attack path
def calc_APV(G_att, G_real, node_ID, attack_cost, attack_detect_prob):
    
    if G_att.nodes[node_ID]["compromised_status"]:
        return 1
    
    att_detect_honey = False
    if G_real.nodes[node_ID]["honeypot"] == 1:
        if random.random() < attack_detect_prob:
            att_detect_honey = True
    elif G_real.nodes[node_ID]["honeypot"] == 2:
        if random.random() < (attack_detect_prob/2):
            att_detect_honey = True
    if att_detect_honey:
#         return (1 - (attack_cost/3) ) * G_real.nodes[node_ID]["normalized_vulnerability"]
        # if the node is honeypot, the APV for honeypot is 0.
        return 0
    else:
        return (1 - (attack_cost/3) ) * G_att.nodes[node_ID]["normalized_vulnerability"]

        


# In[40]:


# Attack Impact by given attack k
# new_compromised_list is new compromised node IDs (do not include already compromised node)
def attack_impact(G, new_compromised_list):
    
    if len(new_compromised_list) == 0:
        return 0
    
    N = G.number_of_nodes()
    
    total_criticality = 0
    for n in new_compromised_list:
        total_criticality += G.nodes[n]["criticality"]
    ai = total_criticality/N
    return ai


# In[ ]:





# In[ ]:





# In[ ]:





# In[41]:


# AS1 – Monitoring attack
# (keep try untile get one)
# return: a dictionary contain all information
def attack_AS_1(G_real, G_att, G_def, node_info_list, monit_time):

    attack_cost = 1
    attack_result = {"attack_cost": attack_cost, "ids": []}
    
    node_id_set = list(G_att.nodes())

    # if all node evicted, do nothing
#     if all(list(nx.get_node_attributes(G_real, "evicted_mark").values())):
#         print("AS_1 Fail")
#         return attack_result
    

    not_get_one = True
    while not_get_one:
        random_id = random.choice(node_id_set)
        # Check if compromised
        if not G_real.nodes[random_id]["evicted_mark"]:
            not_get_one = False
            if random.random() <= G_real.nodes[random_id][
                    "normalized_vulnerability"] * math.exp(-1/monit_time):  # success rate is based on real graph
                node_info_list.append(G_att.nodes[random_id]["id"])
#                 not_get_one = False
                
    return attack_result


# In[1]:


# AS2 – Social engineering
# input: node_info_list: nodes with information collected (dictionary type)
#
# return: a dictionary with "attack cost", and compromised "ids". "ids" is empty if unsuccessful

# TODO: check all attack strategy, if correct vulnerability type is used
def attack_AS_2(node_info_list, G_real, G_att, G_def, P_fake,
                attack_detect_prob, location):
    
    attack_cost = 3

    attack_result = {"attack_cost": attack_cost, "ids": []}

    # fake key probability
    if random.random() <= P_fake:
        if display:
            print("get fake key, failed to compromise")
        return attack_result

    # if outside
    if location is None:
        if node_info_list:
            target_node_id = random.choice(node_info_list)
            if random.random(
            ) < G_real.nodes[target_node_id]["normalized_vulnerability"]:
                attack_result["ids"].append(target_node_id)
                G_real.nodes[target_node_id]["compromised_status"] = True
                G_att.nodes[target_node_id]["compromised_status"] = True
#                 G_def.nodes[target_node_id]["compromised_status"] = True
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

    # collusive attack
    if G_att.nodes[max_APV_id]["compromised_status"] and G_real.nodes[
            max_APV_id]["compromised_status"]:
        attack_result["ids"].append(max_APV_id)
        if display: print("AS2: collusive attack")
        return attack_result

    # compromise attempt
    if random.random() < G_real.nodes[max_APV_id]["normalized_vulnerability"]:
        attack_result["ids"].append(max_APV_id)
        # set it compromised
        G_real.nodes[max_APV_id]["compromised_status"] = True
        G_att.nodes[max_APV_id]["compromised_status"] = True
#         G_def.nodes[max_APV_id]["compromised_status"] = True
    else:
        if display:
            print("AS2: unsuccessful on", max_APV_id, "with vul",
                  G_real.nodes[max_APV_id]["normalized_vulnerability"])

    return attack_result


# In[43]:


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


#     compromised_nodes = []
#     for n in G_att.nodes():
#         if G_real.nodes[n]["compromised_status"]:
#             compromised_nodes.append(n)
    attacked_adjacent = []

    #     all_nodes = list(G_real.nodes())
    for node_id in collection_list:
        if not G_real.nodes[node_id]["evicted_mark"]:
            # if attacker detect deception, use real network
            if random.random() < attack_detect_prob:
                attacked_adjacent += graph_function.adjacent_node(G_real, node_id)
            else:
                attacked_adjacent += graph_function.adjacent_node(G_att, node_id)

    attacked_adjacent = list(set(attacked_adjacent))

    for n in attacked_adjacent:
        if G_att.nodes[n]["compromised_status"] and G_real.nodes[n][
                "compromised_status"]:
            # collusive attack
            attack_result["ids"].append(n)
        else:
            if random.random() <= G_real.nodes[n]["normalized_vulnerability"]:
                G_real.nodes[n]["compromised_status"] = True
                G_att.nodes[n]["compromised_status"] = True
#                 G_def.nodes[n]["compromised_status"] = True
                attack_result["ids"].append(n)

    return attack_result


# In[44]:


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
#         G_def.nodes[max_APV_id]["compromised_status"] = True
    else:
        if display:
            print("AS4: unsuccessful on", max_APV_id, "with vul",
                  G_real.nodes[max_APV_id]["normalized_vulnerability"])
            
    return attack_result



# In[45]:


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
    
    # collusive attack
    if G_att.nodes[max_APV_id]["compromised_status"] and G_real.nodes[
            max_APV_id]["compromised_status"]:
        attack_result["ids"].append(max_APV_id)
        if display: print("AS5: collusive attack")
        return attack_result
    
    #     try compromising
    # if random.uniform(0, 10) <= G_real.nodes[max_APV_id]["normalized_vulnerability"]:
    if random.uniform(0, 10) <= G_real.nodes[max_APV_id]["unknown vulnerability"][0]:
        G_real.nodes[max_APV_id]["compromised_status"] = True
        G_att.nodes[max_APV_id]["compromised_status"] = True
#         G_def.nodes[max_APV_id]["compromised_status"] = True
        attack_result["ids"].append(max_APV_id)
    
    return attack_result


# In[46]:


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
        #         G_def.nodes[max_APV_id]["compromised_status"] = True
        attack_result["ids"].append(max_APV_id)
    else:
        if display:
            print("AS6: unsuccessful on", max_APV_id, "with APV", max_APV)

    return attack_result


# In[47]:


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
#             G_def.nodes[max_APV_id]["compromised_status"] = True
            attack_result["ids"].append(max_APV_id)
            break

    return attack_result


# In[48]:


# AS8 – Data exfiltration
def attack_AS_8(G_real, G_att, G_def, compromised_nodes, attacker_locaton,
                P_fake, attack_detect_prob):

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

    if G_att.nodes[max_APV_id]["compromised_status"] and G_real.nodes[
            max_APV_id]["compromised_status"]:
        # collusive attack
        attack_result["ids"].append(max_APV_id)
    else:
        if random.random() <= G_real.nodes[max_APV_id]["normalized_vulnerability"]:
            # compromise attempt
            G_real.nodes[max_APV_id]["compromised_status"] = True
            G_att.nodes[max_APV_id]["compromised_status"] = True
#             G_def.nodes[max_APV_id]["compromised_status"] = True
            attack_result["ids"].append(max_APV_id)
    
    

    if random.random() <= P_fake:
        if display:
            print("get fake key, failed to compromise")
    else:
        compromised_nodes.append(max_APV_id)

    # data exfiltration
    Thres_c = 30*5 #30  # pre-set value
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


# In[49]:


# below is for class


# In[50]:


def attacker_class_choose_strategy(self, def_strategy_number,
                                   defend_cost_record, defend_impact_record):

    # attacker is 100% sure of CKC subgame
    P_subgame = np.zeros(self.CKC_number + 1)
    P_subgame[self.CKC_position] = 1

    S_j = np.zeros(self.strategy_number)
    for j in range(self.strategy_number):
        for k in range(self.CKC_number + 1):
            S_j[j] += P_subgame[k] * self.prob_believe_opponent[k][j]

    if display: print(f"S_j in att is {S_j}")

    # eq. 19 (Uncertainty g)
    g = self.uncertainty
    
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
    b = 10
    
    # eq. 8
    EU_C = np.zeros(self.strategy_number)
    for i in range(self.strategy_number):
        for j in range(def_strategy_number):
            EU_C[i] += S_j[j] * utility[i, j]
    # Normalization
    if (max(EU_C)-min(EU_C)) != 0:
        EU_C = a + (EU_C-min(EU_C))*(b-a)/(max(EU_C)-min(EU_C))
    self.EU_C = EU_C

    
    # eq. 9
    EU_CMS = np.zeros(self.strategy_number)
    for i in range(self.strategy_number):
        w = np.argmin(utility[i])  # min utility index
        EU_CMS[i] = self.strategy_number * S_j[w] * utility[i][w]
    # Normalization
    if (max(EU_CMS)-min(EU_CMS)) != 0:
        EU_CMS = a + (EU_CMS- min(EU_CMS))*(b-a)/(max(EU_CMS)-min(EU_CMS))
    self.EU_CMS = EU_CMS
    
    # eq. 7
#     HEU = np.zeros(self.strategy_number)
#     for index in range(self.strategy_number):
#         HEU[index] = ((1 - g) * EU_C[index]) + (g * EU_CMS[index])
#     
    if random.random() > g:
        HEU = EU_C
        self.HEU = HEU  # uncertainty case doesn't consider as real HEU
    else:
        HEU = np.ones(self.strategy_number)

    self.AHEU = HEU

    # eq. 23
    AHEU = np.zeros(self.strategy_number)
    for index in range(self.strategy_number):
        AHEU[index] = HEU[index] * self.strat_option[
            self.CKC_position, index]  # for Table 4


        
    self.chosen_strategy = random.choices(range(self.strategy_number),
                                          weights=AHEU,
                                          k=1)[0]
    return self.chosen_strategy


# In[51]:


def attacker_class_execute_strategy(self, G_real, G_def, P_fake,
                                    attack_detect_prob):
    return_value = False
    attack_result = {"attack_cost": 0, "ids": []}

    if self.chosen_strategy == 0:

        attack_result = attack_AS_1(G_real, self.network, G_def,
                                    self.collection_list, self.monit_time)
        return_value = True
    elif self.chosen_strategy == 1:
        attack_result = attack_AS_2(self.collection_list, G_real, self.network,
                                    G_def, P_fake, attack_detect_prob, self.location)
        if attack_result["ids"]:
            self.compromised_nodes.extend(attack_result["ids"])

            #  decrease collection list
            if self.location is None:
                self.collection_list.remove(attack_result["ids"][0])

            return_value = True
        else:
            if display: print("attack 2 failed")

    elif self.chosen_strategy == 2:
        attack_result = attack_AS_3(self.collection_list, G_real, self.network, G_def, P_fake,
                                    attack_detect_prob)
        if attack_result["ids"]:
            self.compromised_nodes.extend(attack_result["ids"])
            return_value = True
        else:
            if display: print("attack 3 failed")

    elif self.chosen_strategy == 3:
        attack_result = attack_AS_4(G_real, self.network, G_def,
                                    attack_detect_prob, P_fake, self.location)
        if attack_result["ids"]:
            self.compromised_nodes.extend(attack_result["ids"])
            return_value = True
        else:
            if display: print("attack 4 failed")

    elif self.chosen_strategy == 4:
        attack_result = attack_AS_5(G_real, self.network, G_def, self.location,
                                    attack_detect_prob)
        if attack_result["ids"]:
            self.compromised_nodes.extend(attack_result["ids"])
            return_value = True
        else:
            if display: print("attack 5 failed")

    elif self.chosen_strategy == 5:
        attack_result = attack_AS_6(G_real, self.network, G_def, self.location,
                                    P_fake, attack_detect_prob)
        if attack_result["ids"]:
            self.compromised_nodes.extend(attack_result["ids"])
            return_value = True
        else:
            if display: print("attack 6 failed")

    elif self.chosen_strategy == 6:
        attack_result = attack_AS_7(G_real, self.network, G_def, self.location,
                                    P_fake, attack_detect_prob)
        if attack_result["ids"]:
            self.compromised_nodes.extend(attack_result["ids"])
            return_value = True
        else:
            if display: print("attack 7 failed")

    else:
        attack_result = attack_AS_8(G_real, self.network, G_def,
                                    self.compromised_nodes, self.location,
                                    P_fake, attack_detect_prob)
        if attack_result["ids"]:
            self.compromised_nodes.extend(attack_result["ids"])
#             return_value = True

        if attack_result["data_exfiltrated"]:
            if display: print("attacker exfiltrate data")
            return_value = True
        else:
            if display: print("attack 8 failed")

    self.impact_record[self.chosen_strategy] = attack_impact(
        G_real, attack_result["ids"])
    
    if attack_result["ids"]:
        self.location = random.choice(attack_result["ids"])
    


    return return_value


# In[53]:


class attacker_class:
    def __init__(self, game, uncertain_scheme, att_detect_UpBod):
        if display: print("create attacker")
        self.network = copy.deepcopy(game.graph.network)  # attacker's view
        self.strategy_number = 8
        self.collection_list = []
        self.location = None
        self.impact_record = np.ones(
            self.strategy_number
        )  # attacker believe all strategy have impact initially
        self.strat_cost = att_strategy_cost(self.strategy_number)
        self.strat_option = att_strategy_option_matrix(
            game.CKC_number, self.strategy_number)  # Table 4
        self.belief_context = [1 /
                               (game.CKC_number + 1)] * (game.CKC_number + 1)
        self.CKC_position = 0
        self.CKC_number = game.CKC_number
        self.prob_believe_opponent = np.zeros(
            (game.CKC_number + 1,
             8))  # 8 is defender strategy number # c_{\kappa}
        self.obs_oppo_strat_history = np.zeros(
            (game.CKC_number + 1, 8))  # 8 is defender strategy number
        self.in_system_time = 1
        self.monit_time = 1
        self.detect_prob = random.uniform(0, att_detect_UpBod)
        self.chosen_strategy = 0
        self.in_honeynet = False
        self.uncertain_scheme = uncertain_scheme
        if self.uncertain_scheme:
            self.uncertainty = 1  #1  # 100% uncertainty at beginning  (scheme change here!)
        else:
            self.uncertainty = 0
        self.HEU = np.zeros(self.strategy_number)
        self.compromised_nodes = []
        self.EU_C = None
        self.EU_CMS = None
        self.AHEU = np.zeros(self.strategy_number)
        self.att_guess_DHEU = np.zeros(self.strategy_number)
        self.chosen_strategy_record = np.zeros(self.strategy_number)
        self.defender_observation = np.zeros(self.strategy_number)
        self.att_guess_def_impact = np.ones(self.strategy_number)
        self.observed_defen_strat = 0
        self.defender_strat_cost = def_strategy_cost(self.strategy_number)

    choose_strategy = attacker_class_choose_strategy

    execute_strategy = attacker_class_execute_strategy

    def reset_in_system_time(self):
        self.in_system_time = 1
        return self.in_system_time

    def next_stage(self):
        if self.CKC_position != 5:
            self.CKC_position += 1

    def reset_attribute(self):
        pass

    def observe_opponent(self, defend_CKC, defen_strategy):
        # Observe strategy
        self.obs_oppo_strat_history[defend_CKC, defen_strategy] += 1
        self.observed_defen_strat = defen_strategy
        self.prob_believe_opponent = update_strategy_probability(
            self.obs_oppo_strat_history)

    def update_attribute(self, dec, _lambda):
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
                                                       self.uncertain_scheme, _lambda)
        # HNE Hitting Ratio
        self.defender_observation[self.chosen_strategy] += 1
        self.att_guess_DHEU = self.att_guess_def_EU_C()

    def att_guess_def_EU_C(self):
        # attacker observe itself
        self.chosen_strategy_record[self.chosen_strategy] += 1

        if np.sum(self.defender_observation) == 0:
            strat_prob = np.zeros(self.strategy_number)
        else:
            strat_prob = self.chosen_strategy_record / np.sum(self.chosen_strategy_record)
        xi = 5

        self.att_guess_def_impact[self.observed_defen_strat] = 1 - self.impact_record[self.chosen_strategy]

        utility = np.zeros((self.strategy_number, self.strategy_number))
        for i in range(self.strategy_number):
            for j in range(self.strategy_number):
                utility[i, j] = (self.att_guess_def_impact[i] +
                                 self.strat_cost[j] / 3) - (self.defender_strat_cost[i] / 3 + self.impact_record[j])
        EU_C = np.zeros(self.strategy_number)
        for i in range(self.strategy_number):
            for j in range(self.strategy_number):
                EU_C[i] += strat_prob[j] * utility[i, j]
        # Normalization
        a = 1
        b = 10
        if (max(EU_C) - min(EU_C)) != 0:
            EU_C = a + (EU_C - min(EU_C)) * (b - a) / (max(EU_C) - min(EU_C))
        self.EU_C = EU_C
        return EU_C


    def random_moving(self):
        if self.location is None:
            return

        neighbor_list = [i for i in self.network[self.location]]
        compromised_neighbor_list = [self.location
                                     ]  # allow attacker stands still
        for index in neighbor_list:
            if self.network.nodes[index]["compromised_status"]:
                compromised_neighbor_list.append(index)

        self.location = random.choice(compromised_neighbor_list)


# In[ ]:





# In[ ]:




