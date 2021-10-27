#!/usr/bin/env python
# coding: utf-8

# In[2]:


import random
import numpy as np
from main import display
import graph_function
import math
import copy
import attacker_function


# In[3]:


def def_strategy_cost(strategy_number):
    defend_cost = np.zeros(strategy_number)
    defend_cost[0] = 1
    defend_cost[1] = 2
    defend_cost[2] = 3
    defend_cost[3] = 3
    defend_cost[4] = 3
    defend_cost[5] = 1
    defend_cost[6] = 2
    defend_cost[7] = 2

    
    return defend_cost


# In[4]:


def defender_uncertainty_update(att_detect, def_monit_time, def_strategy_number, uncertain_scheme, mu):

    # mu = 8 # was 1
    
    uncertainty = 1-math.exp((-mu) *att_detect/def_monit_time)

# (scheme change here!)
    if uncertain_scheme:
        return uncertainty
    else:
        return 0


# In[5]:


def def_strategy_option_matrix(CKC_number, strategy_number, DD_using):
    strat_option = np.zeros(
        (CKC_number + 1, strategy_number))  # last one is full game
    
    if DD_using:
        # For Scheme 1 3 (with DD) 1 3, use below
        # R
        strat_option[0, 0] = 1
        strat_option[0, 7] = 1
        # D
        strat_option[1, 0] = 1
        strat_option[1, 1] = 1
        # E
        strat_option[2, 2] = 1
        strat_option[2, 3] = 1
        strat_option[2, 4] = 1
        strat_option[2, 6] = 1
        # C2
        for i in range(2, strategy_number):
            strat_option[3, i] = 1
        # M
        for i in range(2, strategy_number):
            strat_option[4, i] = 1
        # DE
        for i in range(2, strategy_number):
            strat_option[5, i] = 1
        # Full Game
        for i in range(strategy_number):
            strat_option[6, i] = 1
    else:
        # For Scheme 2 4 (no DD), use below
        # R
        strat_option[0, 0] = 1
        # D
        strat_option[1, 0] = 1
        strat_option[1, 1] = 1
        # E
        strat_option[2, 2] = 1
        strat_option[2, 3] = 1
        # C2
        strat_option[3, 2] = 1
        strat_option[3, 3] = 1
        # M
        strat_option[4, 2] = 1
        strat_option[4, 3] = 1
        # DE
        strat_option[5, 2] = 1
        strat_option[5, 3] = 1
        # Full Game
        strat_option[6, 0] = 1
        strat_option[6, 1] = 1
        strat_option[6, 2] = 1
        strat_option[6, 3] = 1

    return strat_option


# In[6]:


# DS1 – Firewalls
def defense_DS_1(G_real, G_att, G_def):

    for n in G_real.nodes():
        G_real.nodes[n]["unknown vulnerability"][0] = max(
            G_real.nodes[n]["unknown vulnerability"][0] - 0.01, 0)

    for n in G_att.nodes():
        G_att.nodes[n]["unknown vulnerability"][0] = max(
            G_att.nodes[n]["unknown vulnerability"][0] - 0.01, 0)

    for n in G_def.nodes():
        G_def.nodes[n]["unknown vulnerability"][0] = max(
            G_def.nodes[n]["unknown vulnerability"][0] - 0.01, 0)
        
    graph_function.update_vul(G_real)
    graph_function.update_vul(G_att)
    graph_function.update_vul(G_def)


# In[7]:


# DS2 – Patch Management
def defense_DS_2(G_real, G_att, G_def, sv):

    for index in range(sv):
        for n in G_real.nodes():
            G_real.nodes[n]["software vulnerability"][index] = max(
                G_real.nodes[n]["software vulnerability"][index] - 0.01, 0)

        for n in G_real.nodes():
            G_att.nodes[n]["software vulnerability"][index] = max(
                G_att.nodes[n]["software vulnerability"][index] - 0.01, 0)

        for n in G_real.nodes():
            G_def.nodes[n]["software vulnerability"][index] = max(
                G_def.nodes[n]["software vulnerability"][index] - 0.01, 0)
            
    graph_function.update_vul(G_real)
    graph_function.update_vul(G_att)
    graph_function.update_vul(G_def)


# In[8]:


# DS3 – Rekeying Cryptographic Keys
def defense_DS_3(G_real, G_att, G_def, graph):
    
    graph.T_rekey_reset()
    
    graph_function.update_en_vul(G_real, graph.ev, graph.ev_lambda, graph.T_rekey)
    graph_function.update_en_vul(G_att, graph.ev, graph.ev_lambda, graph.T_rekey)
    graph_function.update_en_vul(G_def, graph.ev, graph.ev_lambda, graph.T_rekey)
    
    graph_function.update_vul(G_real)
    graph_function.update_vul(G_att)
    graph_function.update_vul(G_def)


# In[9]:


# DS4 – Eviction
def defense_DS_4(G_real, G_att, G_def, false_neg_prob, false_pos_prob, NIDS_eviction):
    
    for index in list(G_def.nodes()):
        # ignore evicted node for saving time
        if graph_function.is_node_evicted(G_def, index):
            continue
        
        if G_def.nodes[index]["compromised_status"]:
            graph_function.evict_a_node(index, G_real, G_def, G_att)
            if G_real.nodes[index]["compromised_status"]:
                NIDS_eviction[2] += 1
            else:
                NIDS_eviction[3] += 1
#         node_is_compromised = False
#         #         if G_def.has_node(index):
#         if G_def.nodes[index]["compromised_status"]:
#             if random.random() > false_neg_prob:
#                 node_is_compromised = True
#             else:
#                 if display: print("False Negative to compromised node")
#         else:
#             if random.random() < false_pos_prob:
#                 if display: print("False Positive to good node")
#                 node_is_compromised = True

        


#         if G_def.nodes[index]["compromised_status"]:
#             if not graph_function.is_node_evicted(G_def, index):
#                 if random.random() < 1 - false_neg:
#                     graph_function.evict_a_node(index, G_real, G_def, G_att)
#                     counter += 1
#                 else:
#                     if display: print(f"Eviction: False Negative{index}")

#     print(f"DS_4 evict: {counter} node")

#     Detection_rate = 0.9
#     saved_nodes = [];

#     for n in list(G_def.nodes()): # list() avoid "dictionary changed size during iteration Error"
#         if G_def.nodes[n]["compromised_status"]: # defender belived compromised node

#             print("|| did")
#             node_neighbor = list(G_def.neighbors(n))
#             for neighbor_index in node_neighbor:
#                 if G_def.has_edge(n,neighbor_index): G_def.remove_edge(n,neighbor_index)

#             if random.random() < Detection_rate:
#                 print("= did")
#                 for neighbor_index in node_neighbor:
#                     if G_real.has_edge(n,neighbor_index): G_real.remove_edge(n,neighbor_index)
#                     if G_att.has_edge(n,neighbor_index): G_att.remove_edge(n,neighbor_index)
#                 saved_nodes.append(n)
# #             G_def.remove_node(n)  # defender think all compromised node are evicted
# #             if random.random() < Detection_rate:
# #                 G_real.remove_node(n)
# #                 G_att.remove_node(n)
# #                 saved_nodes.append(n)

#     return {"ids": saved_nodes}


# In[10]:


# DS5 – Low/High-Interaction Honeypots
def defense_DS_5(G_real, G_att, G_def, H_G, high_inter, low_inter, inter_per_node):

        
    legitimate_nodes = {}
    for n in G_def.nodes():  # in defender view
        if not graph_function.is_node_evicted(G_def, n):
            legitimate_nodes[n] = G_def.nodes[n]["normalized_vulnerability"]
            
    sorted_node_index = sorted(legitimate_nodes,
                               key=legitimate_nodes.__getitem__,
                               reverse=True)  # sort from high to low

    if len(sorted_node_index) <= (high_inter*inter_per_node + low_inter*inter_per_node):
        if display: print("Not enough node")
        return False

    # add honeypot network to main network
    G_real.add_nodes_from(H_G.nodes(data=True))
    G_real.add_edges_from(H_G.edges(data=True))
    G_att.add_nodes_from(H_G.nodes(data=True))
    G_att.add_edges_from(H_G.edges(data=True))
    G_def.add_nodes_from(H_G.nodes(data=True))
    G_def.add_edges_from(H_G.edges(data=True))
    

    

    # HI to top 15
    counter = 0
    for n in range(high_inter):
        for i in range(inter_per_node):  # 3 per HI honeypot
            G_real.add_edge(sorted_node_index[counter], "HI" + str(n))
            G_att.add_edge(sorted_node_index[counter], "HI" + str(n))
            G_def.add_edge(sorted_node_index[counter], "HI" + str(n))
            counter += 1
    # LI to next top 30
    for n in range(low_inter):
        for i in range(inter_per_node):  # 3 per LI honeypot
            G_real.add_edge(sorted_node_index[counter], "LI" + str(n))
            G_att.add_edge(sorted_node_index[counter], "LI" + str(n))
            G_def.add_edge(sorted_node_index[counter], "LI" + str(n))
            counter += 1
    return True
# Update network attribute


# In[1]:


# DS6 – Honey information
# this defend randomly decrease vulnerability in attacker view


def defense_DS_6(G_att, sv, ev, graph):

    for n in G_att.nodes():
        # change software vulnerability
        for sv_index in range(sv):
            decrease_value = random.uniform(
                0, G_att.nodes[n]["software vulnerability"][sv_index])
            G_att.nodes[n]["software vulnerability"][
                sv_index] -= decrease_value

        # change encryption vulnerability
        for ev_index in range(ev):
            decrease_value = random.uniform(
                0,
                G_att.nodes[n]["original_encryption_vulnerability"][sv_index])
            G_att.nodes[n]["original_encryption_vulnerability"][
                ev_index] -= decrease_value

        # change unknown vulnerability
        decrease_value = random.uniform(
            0, G_att.nodes[n]["unknown vulnerability"][0])
        G_att.nodes[n]["unknown vulnerability"][0] -= decrease_value

    graph_function.update_en_vul(G_att, graph.ev, graph.ev_lambda, graph.T_rekey)
    graph_function.update_vul(G_att)


# In[12]:


# DS7 – Fake keys
# update for AS using encryption vul.
def defense_DS_7(P_fake, att_detect):
    
#     P_fake = 0.5 - att_detect
    P_fake = 1 - att_detect


# In[13]:


# DS8 – Hiding network topology edges
# randomly pick a node, and remove the edge connect to highest criticality node among the node nieghbor
def defense_DS_8(G_real, G_att, G_def, using_honeynet, low_inter, high_inter, inter_per_node):
    hide_edge_rate = 0.2
    edge_number = graph_function.number_of_edge_without_honeypot(G_att)
    to_remove_edge_number = int(round(edge_number*hide_edge_rate))
    if display: print("hide "+str(to_remove_edge_number)+" edges")
    
#     node_id_set = list(G_att.nodes(data=True))
    node_id_set = graph_function.ids_without_honeypot(G_att)
    
    while to_remove_edge_number!=0:
#         if display: print(f"Hide remain {to_remove_edge_number}")
        chosen_node = random.choice(node_id_set) 
        
        # get max adjacent node
        max_criticality_id = None
        max_criticality_value = 0
        for neighbor_id in G_att.neighbors(chosen_node):
            if G_att.nodes[neighbor_id]["honeypot"] == 0: # do not hide edge to honeynet
                if G_att.nodes[neighbor_id]["criticality"] >= max_criticality_value:
                    max_criticality_id = neighbor_id
                    max_criticality_value = G_att.nodes[neighbor_id]["criticality"]

        # Hide edge
        if max_criticality_id != None:
            G_att.remove_edge(chosen_node, max_criticality_id)
            to_remove_edge_number-=1
#             if display: print("hide edge ["+str(chosen_node)+" , "+str(max_criticality_id)+"]")
        
        
        # stop loop if no removable edge
        if using_honeynet:
            honey_node_number = low_inter + high_inter
            complete_honeynet_edge_number = (honey_node_number*(honey_node_number-1))/2 # edge number in complete graph
            network_to_honeynet_edge_number = honey_node_number*inter_per_node
            if G_att.number_of_edges() <= (network_to_honeynet_edge_number+complete_honeynet_edge_number):
                if display: print("All edge left is related to Honeynet")
                to_remove_edge_number = 0
        else:
            # avoid no edge error
            if G_att.number_of_edges() == 0:
                if display: print("There are no more edges.")
                to_remove_edge_number = 0
                    
        
        
        
        


# In[ ]:





# In[ ]:





# In[2]:


def defender_class_choose_strategy(self, att_choose_strategy, att_strategy_number,
                                   attack_cost_record, attack_impact_record):
    
    S_j = np.zeros(self.strategy_number)
    for j in range(self.strategy_number):
        for k in range(self.CKC_number + 1):
            S_j[j] += self.P_subgame[k] * self.prob_believe_opponent[k][j]

    if display: print(f"S_j in def is {S_j}, sum is {sum(S_j)}")

    # eq. 19 (Uncertainty g)
    g = self.uncertainty

    # Update defense impact
    self.impact_record[self.chosen_strategy] = 1 - attack_impact_record[att_choose_strategy]

    # eq. 17
    utility = np.zeros((self.strategy_number, att_strategy_number))
    for i in range(self.strategy_number):
        for j in range(att_strategy_number):
            utility[i,
                    j] = (self.impact_record[i] +
                          attack_cost_record[j] / 3) - (
                              self.strat_cost[i] / 3 + attack_impact_record[j])
            
    # normalization range
    a = 1
    b = 10
    
    # eq. 8
    EU_C = np.zeros(self.strategy_number)
    for i in range(self.strategy_number):
        for j in range(att_strategy_number):
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
    HEU = EU_C
    self.HEU = HEU  # uncertainty case doesn't consider as real HEU
#     if random.random() > g:
        
#     else:
#         HEU = np.ones(self.strategy_number)


    # eq. 23
    DHEU = np.zeros(self.strategy_number)
    if random.random() > g:
        self.DHEU = HEU
        for index in range(self.strategy_number):
            DHEU[index] = HEU[index] * self.strat_option[
                self.CKC_position, index]  # for Table 4
    else:
        HEU = np.ones(self.strategy_number)
        self.DHEU = HEU
        for index in range(self.strategy_number):
            DHEU[index] = HEU[index] * self.strat_option[
                6, index]  # for Table 4
    
    def_chosen_strategy = random.choices(range(self.strategy_number),
                                         weights=DHEU,
                                         k=1)[0]
    
    self.chosen_strategy = def_chosen_strategy
    if self.chosen_strategy == 4 or self.chosen_strategy == 5 or self.chosen_strategy == 6 or self.chosen_strategy == 7:
        self.deception_tech_used = True
        self.dec = def_strategy_cost(self.strategy_number)[self.chosen_strategy]
    else:
        self.dec = 0

    return def_chosen_strategy



# In[15]:


def defender_class_execute_strategy(self, G_att, att_detect, graph, FNR, FPR, NIDS_eviction):

    return_value = False
    if self.chosen_strategy == 0:
        defense_DS_1(graph.network, G_att, self.network)
        return_value = True
    elif self.chosen_strategy == 1:
        defense_DS_2(graph.network, G_att, self.network, graph.sv)
        return_value = True
    elif self.chosen_strategy == 2:
        defense_DS_3(graph.network, G_att, self.network, graph)
        return_value = True
    elif self.chosen_strategy == 3:
        defense_DS_4(graph.network, G_att, self.network, FNR, FPR, NIDS_eviction)
        return_value = True
    elif self.chosen_strategy == 4:
        graph.new_honeypot()
        # use this strategy only once in a simulation
        if not graph.using_honeynet:
            strat_success = defense_DS_5(graph.network, G_att, self.network,
                                         graph.honey_net, graph.high_inter,
                                         graph.low_inter, graph.inter_per_node)
            if strat_success:
                graph.using_honeynet = True
                return_value = True
        else:
            return_value = False
    elif self.chosen_strategy == 5:
        defense_DS_6(G_att, graph.sv, graph.ev, graph)
        return_value = True
    elif self.chosen_strategy == 6:
        defense_DS_7(self.P_fake, att_detect)
        return_value = True
    elif self.chosen_strategy == 7:
        defense_DS_8(graph.network, G_att, self.network, graph.using_honeynet,
                     graph.low_inter, graph.high_inter, graph.inter_per_node)
        return_value = True

    return return_value


# In[16]:


class defender_class:
    def __init__(self, game, uncertain_scheme):
        if display: print("create defender")
        self.network = copy.deepcopy(game.graph.network)  # defender's view
        self.strategy_number = 8
        self.key_time = 1
        self.monit_time = 1
        self.dec = 0
        self.strat_cost = def_strategy_cost(self.strategy_number)
        self.impact_record = np.ones(self.strategy_number)
        self.P_fake = 0  # fake key for DS7
        self.CKC_position = 6  # 6 means full game
        self.CKC_number = game.CKC_number
        self.strat_option = def_strategy_option_matrix(
            game.CKC_number, self.strategy_number, game.DD_using)  # Table 4
        self.chosen_strategy = random.randint(0, 7)
        self.prob_believe_opponent = np.zeros(
            (game.CKC_number + 1, game.attacker.strategy_number))
        self.obs_oppo_strat_history = np.zeros(
            (game.CKC_number + 1, game.attacker.strategy_number))
        self.P_subgame = np.zeros(game.CKC_number + 1)  # Eq. 5: belief context
        self.subgrame_history = np.zeros(game.CKC_number + 1)
        self.deception_tech_used = False
        self.uncertain_scheme = uncertain_scheme
        if self.uncertain_scheme:
            self.uncertainty = 1  #1 # 100% uncertainty at beginning  (scheme change here!)
        else:
            self.uncertainty = 0
        self.HEU = np.zeros(self.strategy_number)
        self.EU_C = None
        self.EU_CMS = None
        self.DHEU = np.zeros(self.strategy_number)
        self.def_guess_AHEU = np.zeros(self.strategy_number)
        self.chosen_strategy_record = np.zeros(self.strategy_number)
        self.attacker_observation = np.zeros(self.strategy_number)
        self.attacker_strat_cost = attacker_function.att_strategy_cost(self.strategy_number)

        
    def observe_opponent(self, attack_impact_record, attack_CKC,
                         attack_strategy):
        # Observe strategy
        self.obs_oppo_strat_history[attack_CKC, attack_strategy] += 1
        self.prob_believe_opponent = graph_function.update_strategy_probability(
            self.obs_oppo_strat_history)

        # belief context
        self.subgrame_history[attack_CKC] += 1
        self.P_subgame = self.subgrame_history / (sum(self.subgrame_history))

    def update_attribute(self, att_detect, mu, attack_impact):
        # key_time
        self.key_time += 1
        # monitor time
        self.monit_time += 1
        # uncertainty
        self.uncertainty = defender_uncertainty_update(att_detect,
                                                       self.monit_time,
                                                       self.strategy_number,
                                                       self.uncertain_scheme, mu)
        self.def_guess_AHEU = self.def_guess_att_EU_C(attack_impact)
        self.attacker_observation[self.chosen_strategy] += 1

    def def_guess_att_EU_C(self, attack_impact):
        # defender observe itself
        self.chosen_strategy_record[self.chosen_strategy] += 1

        if np.sum(self.attacker_observation) == 0:
            strat_prob = np.zeros(self.strategy_number)
        else:
            strat_prob = self.chosen_strategy_record / np.sum(self.chosen_strategy_record)

        utility = np.zeros((self.strategy_number, self.strategy_number))
        for i in range(self.strategy_number):
            for j in range(self.strategy_number):
                utility[i, j] = (attack_impact[i] +
                                 self.strat_cost[j] / 3) - (self.attacker_strat_cost[i] / 3 + self.impact_record[j])
        EU_C = np.zeros(self.strategy_number)
        for i in range(self.strategy_number):
            for j in range(self.strategy_number):
                EU_C[i] += strat_prob[j] * utility[i, j]
        # Normalization
        a = 1
        b = 10
        if (max(EU_C) - min(EU_C)) != 0:
            EU_C = a + (EU_C - min(EU_C)) * (b - a) / (max(EU_C) - min(EU_C))
        return EU_C

    def reset_attribute(self, attack_impact_record, CKC_number):
        self.key_time = 1
        self.monit_time = 1
        self.P_fake = 0
        self.impact_record = np.ones(self.strategy_number)
        self.belief_context = [1 / (CKC_number + 1)] * (CKC_number + 1)
        self.obs_oppo_strat_history = np.zeros(
            (CKC_number + 1, self.strategy_number))
        self.dec = 0
        self.deception_tech_used = False
        if self.uncertain_scheme:
            self.uncertainty = 1  #(scheme change here!)
        else:
            self.uncertainty = 0

    choose_strategy = defender_class_choose_strategy
    execute_strategy = defender_class_execute_strategy

    def decide_CKC_posi(self, att_detect, att_CKC_stage):
        g = self.uncertainty
        if random.random() > g:
            self.CKC_position = att_CKC_stage
            return True
        else:
            self.CKC_position = 6  # full game
            return False


# In[ ]:





# In[ ]:





# In[ ]:




