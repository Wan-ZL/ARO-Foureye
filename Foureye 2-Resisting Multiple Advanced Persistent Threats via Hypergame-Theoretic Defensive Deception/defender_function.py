#!/usr/bin/env python
# coding: utf-8

# In[29]:


from main import display
# from graph_function import *
import graph_function
import attacker_function
import concurrent
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import copy
import time
import pickle
import itertools
import sklearn
# from pure_sklearn.map import convert_estimator

# In[30]:


def def_strategy_cost(strategy_number):
    defend_cost = np.zeros(strategy_number)
    defend_cost[0] = 1
    defend_cost[1] = 2
    defend_cost[2] = 2
    defend_cost[3] = 2 # was 3
    defend_cost[4] = 3
    defend_cost[5] = 1
    defend_cost[6] = 2
    defend_cost[7] = 2
    if strategy_number-1 == 8: defend_cost[8] = 0
    # defend_cost = np.ones(strategy_number)          # test
    return defend_cost


# In[31]:


def defender_uncertainty_update(att_detect, def_monit_time, uncertain_scheme, decision_scheme):
    mu = 10  # was 1

    uncertainty = 1 - math.exp((-mu) * att_detect / def_monit_time)
    if decision_scheme == 0:
        return 1
    else:
        if uncertain_scheme:
            return uncertainty     # for test. orignial: uncertainty
        else:
            return 0



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


# In[34]:


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


# In[35]:


# DS3 – Rekeying Cryptographic Keys
def defense_DS_3(G_real, G_att, G_def, graph):
    graph.T_rekey_reset()

    graph_function.update_en_vul(G_real, graph.ev, graph.ev_lambda, graph.T_rekey)
    graph_function.update_en_vul(G_att, graph.ev, graph.ev_lambda, graph.T_rekey)
    graph_function.update_en_vul(G_def, graph.ev, graph.ev_lambda, graph.T_rekey)

    graph_function.update_vul(G_real)
    graph_function.update_vul(G_att)
    graph_function.update_vul(G_def)


# In[36]:


# DS4 – Eviction
def defense_DS_4(G_real, G_att_list, G_def):
    all_compromised_node = []
    all_importance = []
    eviction_percentage = 0.25

    for index in list(G_def.nodes()):
        # ignore evicted node for saving time
        if graph_function.is_node_evicted(G_def, index):
            continue

        if G_def.nodes[index]["compromised_status"]:
            all_compromised_node.append(index)
            all_importance.append(G_def.nodes[index]["importance"])
            # graph_function.evict_a_node(index, G_real, G_def, G_att_list)

    # sort node based on importance
    sorted_compromised_node = [node_index for _,node_index in sorted(zip(all_importance, all_compromised_node))]
    top_sorted_compromised_node = sorted_compromised_node[-int(len(sorted_compromised_node)*eviction_percentage):]


    # evict top compromised nodes
    for index in top_sorted_compromised_node:
        graph_function.evict_a_node(index, G_real, G_def, G_att_list)



# DS5 – Low/High-Interaction Honeypots
def defense_DS_5(G_real, G_att_list, G_def, H_G, high_inter, low_inter, inter_per_node):
    legitimate_nodes = {}
    for n in G_def.nodes():  # in defender view
        if not graph_function.is_node_evicted(G_def, n):
            legitimate_nodes[n] = G_def.nodes[n]["normalized_vulnerability"]

    sorted_node_index = sorted(legitimate_nodes,
                               key=legitimate_nodes.__getitem__,
                               reverse=True)  # sort from high to low

    if len(sorted_node_index) <= (high_inter * inter_per_node + low_inter * inter_per_node):
        if display:
            print("Not enough node")
        return False

    # add honeypot network to main network
    G_real.add_nodes_from(H_G.nodes(data=True))
    G_real.add_edges_from(H_G.edges(data=True))
    G_def.add_nodes_from(H_G.nodes(data=True))
    G_def.add_edges_from(H_G.edges(data=True))
    for G_att in G_att_list:
        G_att.add_nodes_from(H_G.nodes(data=True))
        G_att.add_edges_from(H_G.edges(data=True))

    # HI to top 15
    counter = 0
    for n in range(high_inter):
        for i in range(inter_per_node):  # 3 per HI honeypot
            G_real.add_edge(sorted_node_index[counter], "HI" + str(n))
            G_def.add_edge(sorted_node_index[counter], "HI" + str(n))
            for G_att in G_att_list:
                G_att.add_edge(sorted_node_index[counter], "HI" + str(n))
            counter += 1
    # LI to next top 30
    for n in range(low_inter):
        for i in range(inter_per_node):  # 3 per LI honeypot
            G_real.add_edge(sorted_node_index[counter], "LI" + str(n))
            G_def.add_edge(sorted_node_index[counter], "LI" + str(n))
            for G_att in G_att_list:
                G_att.add_edge(sorted_node_index[counter], "LI" + str(n))
            counter += 1

    return True


# Update network attribute


# DS6 – Honey information
# this defend randomly decrease vulnerability in attacker view
def defense_DS_6(G_att_list, sv, ev, graph):
    if not G_att_list:
        print("DS6: No attacker in system")
        return

    for n in G_att_list[0].nodes():
        # change software vulnerability
        for sv_index in range(sv):
            decrease_value = random.uniform(
                0, G_att_list[0].nodes[n]["software vulnerability"][sv_index])
            # for all attacker
            for G_att in G_att_list:
                G_att.nodes[n]["software vulnerability"][sv_index] -= decrease_value

        # change encryption vulnerability
        for ev_index in range(ev):
            decrease_value = random.uniform(0, G_att_list[0].nodes[n]["original_encryption_vulnerability"][sv_index])
            # for all attacker
            for G_att in G_att_list:
                G_att.nodes[n]["original_encryption_vulnerability"][ev_index] -= decrease_value

        # change unknown vulnerability
        decrease_value = random.uniform(0, G_att_list[0].nodes[n]["unknown vulnerability"][0])
        # for all attacker
        for G_att in G_att_list:
            G_att.nodes[n]["unknown vulnerability"][0] -= decrease_value
    # for all attacker
    for G_att in G_att_list:
        graph_function.update_en_vul(G_att, graph.ev, graph.ev_lambda, graph.T_rekey)
        graph_function.update_vul(G_att)


# DS7 – Fake keys
# update for AS using encryption vul.
def defense_DS_7(P_fake_list, att_detect_list):
    for index in range(len(P_fake_list)):
        P_fake_list[index][0] = 1 - att_detect_list[index]


# In[40]:


# DS8 – Hiding network topology edges
# randomly pick a node, and remove the edge connect to highest criticality node among the node nieghbor
def defense_DS_8(G_real, G_att_list, G_def, using_honeynet, low_inter, high_inter, inter_per_node):
    if not G_att_list:  # if list empty
        print("DS8: No attacker in system")
        return

    hide_edge_rate = 0.2
    edge_number = graph_function.number_of_edge_without_honeypot(G_att_list[0])
    to_remove_edge_number = int(round(edge_number * hide_edge_rate))
    if display: print("hide " + str(to_remove_edge_number) + " edges")

    #     node_id_set = list(G_att.nodes(data=True))
    node_id_set = graph_function.ids_without_honeypot(G_att_list[0])

    while to_remove_edge_number != 0:
        #         if display: print(f"Hide remain {to_remove_edge_number}")
        chosen_node = random.choice(node_id_set)

        # get max adjacent node
        max_criticality_id = None
        max_criticality_value = 0
        for neighbor_id in G_att_list[0].neighbors(chosen_node):
            if G_att_list[0].nodes[neighbor_id]["honeypot"] == 0:  # do not hide edge to honeynet
                if G_att_list[0].nodes[neighbor_id]["criticality"] >= max_criticality_value:
                    max_criticality_id = neighbor_id
                    max_criticality_value = G_att_list[0].nodes[neighbor_id]["criticality"]

        # Hide edge
        if max_criticality_id != None:
            for G_att in G_att_list:
                if G_att.has_edge(max_criticality_id, chosen_node):
                    G_att.remove_edge(chosen_node, max_criticality_id)
            to_remove_edge_number -= 1
        #             if display: print("hide edge ["+str(chosen_node)+" , "+str(max_criticality_id)+"]")

        # stop loop if no removable edge
        if using_honeynet:
            honey_node_number = low_inter + high_inter
            complete_honeynet_edge_number = (honey_node_number * (
                    honey_node_number - 1)) / 2  # edge number in complete graph
            network_to_honeynet_edge_number = honey_node_number * inter_per_node
            if G_att_list[0].number_of_edges() <= (network_to_honeynet_edge_number + complete_honeynet_edge_number):
                if display: print("All edge left is related to Honeynet")
                to_remove_edge_number = 0
        else:
            # avoid no edge error
            if G_att_list[0].number_of_edges() == 0:
                if display: print("There are no more edges.")
                to_remove_edge_number = 0


# In[ ]:
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


# In[41]:


def defender_class_create_bundle(self, DD_using):
    strategy_option = np.zeros(self.strategy_number)

    if DD_using:
        # For Scheme 1 3 (with DD) 1 3, use below
        for def_CKC in self.CKC_list:
            if def_CKC == 0:
                # R
                strategy_option[0] = 1
                strategy_option[7] = 1
                if len(strategy_option)-1 == 8: strategy_option[8] = 1
            elif def_CKC == 1:
                # D
                strategy_option[0] = 1
                strategy_option[1] = 1
                strategy_option[7] = 1
                if len(strategy_option)-1 == 8: strategy_option[8] = 1
            elif def_CKC == 2:
                # E
                strategy_option[1] = 1
                strategy_option[2] = 1
                strategy_option[3] = 1    # test feature: orginin: remove comment
                strategy_option[4] = 1
                strategy_option[6] = 1
                strategy_option[7] = 1
                if len(strategy_option)-1 == 8: strategy_option[8] = 1
            elif def_CKC == 3:
                # C2
                for i in range(1, self.strategy_number):
                    strategy_option[i] = 1
                # strategy_option[3] = 0  # test feature
            elif def_CKC == 4:
                # M
                for i in range(1, self.strategy_number):
                    strategy_option[i] = 1
                # strategy_option[3] = 0  # test feature
            elif def_CKC == 5:
                # DE
                for i in range(1, self.strategy_number):
                    strategy_option[i] = 1
            else:
                print(f"def_CKC {def_CKC}")
                raise Exception("def_CKC is more than 5")

        if len(self.CKC_list) == 0:
            # Full GAME
            for i in range(self.strategy_number):
                strategy_option[i] = 1
    else:
        # For Scheme 2 4 (no DD), use below
        for def_CKC in self.CKC_list:
            if def_CKC == 0:
                # R
                strategy_option[0] = 1
                if len(strategy_option)-1 == 8: strategy_option[8] = 1
            elif def_CKC == 1:
                # D
                strategy_option[0] = 1
                strategy_option[1] = 1
                if len(strategy_option)-1 == 8: strategy_option[8] = 1
            elif def_CKC == 2:
                # E
                strategy_option[1] = 1
                strategy_option[2] = 1
                strategy_option[3] = 1    # test feature: orginin: remove comment
                if len(strategy_option)-1 == 8: strategy_option[8] = 1
            elif def_CKC == 3:
                # C2
                strategy_option[1] = 1
                strategy_option[2] = 1
                strategy_option[3] = 1    # test feature: orginin: remove comment
                if len(strategy_option)-1 == 8: strategy_option[8] = 1
            elif def_CKC == 4:
                # M
                strategy_option[1] = 1
                strategy_option[2] = 1
                strategy_option[3] = 1    # test feature: orginin: remove comment
                if len(strategy_option)-1 == 8: strategy_option[8] = 1
            elif def_CKC == 5:
                # DE
                strategy_option[1] = 1
                strategy_option[2] = 1
                strategy_option[3] = 1
                if len(strategy_option)-1 == 8: strategy_option[8] = 1
            else:
                # Full GAME
                print(f"def_CKC {def_CKC}")
                raise Exception("def_CKC is more than 5")

        if len(self.CKC_list) == 0:
            # Full GAME
            strategy_option[0] = 1
            strategy_option[1] = 1
            strategy_option[2] = 1
            strategy_option[3] = 1
            if len(strategy_option)-1 == 8: strategy_option[8] = 1

    if self.use_bundle:
        # strategy option to ID list
        strategy_avaliable_ID_list = [
            i for i, x in enumerate(strategy_option) if x == 1]

        # create all possible bundle
        all_strategy_bundle = []
        for length in range(1, len(strategy_avaliable_ID_list)+1):
            all_strategy_bundle.extend(
                list(itertools.combinations(strategy_avaliable_ID_list, length)))
        # tuple to list
        all_strategy_bundle = [list(bundle) for bundle in all_strategy_bundle]


        # get all possible bundles that are not out of budget
        optional_strategy_bundle_list = []
        for bundle in all_strategy_bundle:
            bundle_cost = 0
            for strategy_id in bundle:
                bundle_cost += self.strat_cost[strategy_id]
            if bundle_cost <= self.cost_limit:
                optional_strategy_bundle_list.append(bundle)

        self.optional_bundle_list = optional_strategy_bundle_list
    else:
        self.optional_bundle_list = [[i] for i, x in enumerate(strategy_option) if x == 1]


# In[42]:


def defender_class_choose_bundle(self, att_strategy_number, attack_cost_record, attack_impact_record, vary_name, vary_value, att_overall_impact, attacker_list):
    S_j = np.ones(self.strategy_number) / self.strategy_number  # make sure the sum of S_j is 1
    c_kj = _2darray_normalization(self.observed_strategy_count)
    p_k = array_normalization(self.observed_CKC_count)
    for j in range(len(S_j)):
        temp_S = 0
        for k in range(len(p_k)):
            temp_S += p_k[k] * c_kj[k][j]
        S_j[j] = temp_S
    # save data for ML prediction
    self.s_j_window_record = np.vstack([self.s_j_window_record, S_j])
    self.s_j_window_record = np.delete(self.s_j_window_record, 0, 0)

    # flip coin of uncertainty

    # certain level
    # if self.decision_scheme == 1:
    #     pass
        # c_kj = _2darray_normalization(self.observed_strategy_count)
        # p_k = array_normalization(self.observed_CKC_count)
        # for j in range(len(S_j)):
        #     temp_S = 0
        #     for k in range(len(p_k)):
        #         temp_S += p_k[k] * c_kj[k][j]
        #     S_j[j] = temp_S
    # if self.decision_scheme == 2:
    #     # ======= predict strategy separately =======
    #     # use trained model
    #     if vary_name is None:
    #         # fixed setting
    #         if self.uncertain_scheme:  # when consider uncertianty
    #             # print("using IPI")
    #             the_file = open("data/trained_ML_model_list/knn_trained_model_DD-IPI.pkl", "rb")
    #         else:  # when not consider uncertainty
    #             the_file = open("data/trained_ML_model_list/knn_trained_model_DD-PI.pkl", "rb")
    #     else:
    #         # varying settings
    #         if self.uncertain_scheme:  # when consider uncertianty
    #             # print("using IPI")
    #             the_file = open("data_vary/"+vary_name+"="+str(vary_value)+"/trained_ML_model_list/knn_trained_model_DD-IPI.pkl", "rb")
    #         else:  # when not consider uncertainty
    #             the_file = open("data_vary/"+vary_name+"="+str(vary_value)+"/trained_ML_model_list/knn_trained_model_DD-PI.pkl", "rb")
    #
    #     knn_model_list = pickle.load(the_file)
    #     the_file.close()
    #     y_pred = np.zeros(self.strategy_number)
    #     s_j_window_record_normalized = _2darray_normalization(self.s_j_window_record)
    #     for index in range(len(knn_model_list)):
    #         y_pred[index] = knn_model_list[index].predict(s_j_window_record_normalized[:, index].reshape(1, -1))
    #
    #     # print(y_pred)
    #     if sum(y_pred) != 0:  # divide zero would get 'nan'
    #         S_j = y_pred / sum(y_pred)  # normalization
    #     # ===========================================

    # eq. 17
    utility = np.zeros((self.strategy_number, att_strategy_number))
    for i in range(self.strategy_number):
        for j in range(att_strategy_number):
            utility[i,
                    j] = (self.impact_record[i] +
                          attack_cost_record[j] / 3) - (
                                 self.strat_cost[i] / 3 + attack_impact_record[j])

    # eq. 8
    EU_C = np.zeros(self.strategy_number)
    for i in range(self.strategy_number):
        for j in range(att_strategy_number):
            EU_C[i] += S_j[j] * utility[i, j]


    # eq. 9
    EU_CMS = np.zeros(self.strategy_number)
    for i in range(self.strategy_number):
        w = np.argmin(utility[i])  # min utility index
        EU_CMS[i] = self.strategy_number * S_j[w] * utility[i][w]


    HEU = EU_C

    # ================ above is useless ================
    DHEU = np.zeros(self.strategy_number)
    for attacker in attacker_list:
        DHEU += attacker.defender_HEU

    # normalization range
    a = 1
    b = 9
    # Min-Max Normalization
    if (max(DHEU) - min(DHEU)) != 0:
        DHEU = a + (DHEU - min(DHEU)) * (b - a) / (max(DHEU) - min(DHEU))
    else:
        DHEU = np.ones(self.strategy_number) * a
    self.DHEU = DHEU  # uncertainty case doesn't consider as real DHEU

    bundle_DHEU = []
    bundle_lambda = 2

    for bundle in self.optional_bundle_list:
        C_DHEU_value = 0
        for strategy_id in bundle:
            C_DHEU_value += DHEU[strategy_id]
        # C_DHEU_value = C_DHEU_value * math.exp(-bundle_lambda * len(bundle))
        bundle_DHEU.append(C_DHEU_value)

    if random.random() < self.uncertainty:
        # uncertain level
        bundle_DHEU = np.zeros(len(self.optional_bundle_list))

    # save for training ML model
    self.S_j = S_j

    if self.decision_scheme == 2:
        # use trained model
        if vary_name is None:
            # fixed setting
            if self.uncertain_scheme:  # when consider uncertianty
                # print("using IPI")
                the_file = open("data/trained_ML_model/trained_classi_model_ML_collect_data_IPI.pkl", "rb")
            else:  # when not consider uncertainty
                the_file = open("data/trained_ML_model/trained_classi_model_ML_collect_data_PI.pkl", "rb")
        else:
            # varying settings
            if self.uncertain_scheme:  # when consider uncertianty
                the_file = open("data_vary/"+vary_name+"="+str(vary_value)+"/trained_ML_model/trained_classi_model_ML_collect_data_IPI.pkl", "rb")
            else:  # when not consider uncertainty
                the_file = open("data_vary/"+vary_name+"="+str(vary_value)+"/trained_ML_model/trained_classi_model_ML_collect_data_PI.pkl", "rb")

        ML_model = pickle.load(the_file)
        the_file.close()
        data_x = []
        data_x = np.concatenate((data_x, self.att_previous_strat))
        data_x = np.concatenate((data_x, attack_cost_record))
        data_x = np.concatenate((data_x, self.strat_cost))
        data_x = np.concatenate((data_x, attack_impact_record))  # from index 24 to 31
        data_x = np.concatenate((data_x, [att_overall_impact]))  # index 32
        data_x = np.concatenate((data_x, self.impact_record))  # from index 33 to 40
        data_x = np.concatenate((data_x, [self.uncertainty]))
        data_x = np.concatenate((data_x, self.att_previous_CKC))
        # y_pred = ML_model.predict(data_x.reshape(1, -1))

        # ML create bundle strategy
        y_pred_index = ML_model.classes_
        [y_pred_1] = ML_model.predict_proba(data_x.reshape(1, -1))

        y_pred_dic = {y_pred_index[i]: y_pred_1[i] for i in range(len(y_pred_index))}

        temp_dict = list(y_pred_dic.items())                # shuffle dictionary
        random.shuffle(temp_dict)
        y_pred_dic = dict(temp_dict)

        # # new bundle create code
        # bundle_cost = 0
        # ml_bundle = []
        # while bundle_cost <= self.cost_limit:
        #     max_opt = max(y_pred_dic, key=y_pred_dic.get)     # get an optimal defense strategy
        #     if bundle_cost + self.strat_cost[max_opt] <= self.cost_limit:
        #         ml_bundle.append(max_opt)       # add to bundle
        #         y_pred_dic.pop(max_opt)         # avoid duplicate chosen
        #     else:
        #         break

        # old bundle create code
        # since the bundle budget limit is 3, it's impossible to have 3 strategies in bundle. So two options are enough.
        first_opt = max(y_pred_dic, key=y_pred_dic.get)     # get first optimal defense strategy
        # print(f"first max: {first_opt}")
        y_pred_dic.pop(first_opt)
        second_opt = max(y_pred_dic, key=y_pred_dic.get)    # get second optimal defense strategy
        ml_bundle = [first_opt]
        if self.strat_cost[first_opt] + self.strat_cost[second_opt] <= self.cost_limit and first_opt != second_opt and second_opt !=3: # DS4 causes system fail early
            ml_bundle.append(second_opt)

        self.chosen_strategy_list = ml_bundle
    else:
        # Selection Scheme
        if self.decision_scheme == 0 or self.decision_scheme == 3:  # for random selection scheme
            self.chosen_strategy_list = random.choices(self.optional_bundle_list)[0]
        elif self.decision_scheme == 1 or self.decision_scheme == 2:  # DHEU-based selection scheme
            if sum(bundle_DHEU) == 0:  # fix python 3.5 error
                self.chosen_strategy_list = random.choices(self.optional_bundle_list)[0]
            else:
                self.chosen_strategy_list = random.choices(self.optional_bundle_list, weights=bundle_DHEU)[0]
        else:
            raise Exception("Error: Unknown decision_scheme")

    # update 'dec'
    total_cost = 0 #{1,2,3}
    self.dec = 0
    for chosen_strategy in self.chosen_strategy_list:
        if 4 <= chosen_strategy <=7:
            self.dec = 1
            break



def defender_class_execute_strategy(self, G_att_list, att_detect_list, graph, FNR, FPR, NIDS_eviction, P_fake_list):
    return_value = False

    for chosen_strategy in self.chosen_strategy_list:
        if chosen_strategy == 0:
            for G_att in G_att_list:
                defense_DS_1(graph.network, G_att, self.network)
            return_value = True
        elif chosen_strategy == 1:
            for G_att in G_att_list:
                defense_DS_2(graph.network, G_att, self.network, graph.sv)
            return_value = True
        elif chosen_strategy == 2:
            for G_att in G_att_list:
                defense_DS_3(graph.network, G_att, self.network, graph)
            return_value = True
        elif chosen_strategy == 3:
            defense_DS_4(graph.network, G_att_list, self.network)
            return_value = True
        elif chosen_strategy == 4:
            graph.new_honeypot()
            # use this strategy only once in a simulation
            if not graph.using_honeynet:
                strat_success = defense_DS_5(graph.network, G_att_list, self.network,
                                             graph.honey_net, graph.high_inter,
                                             graph.low_inter, graph.inter_per_node)
                if strat_success:
                    graph.using_honeynet = True
                    return_value = True
                    # graph_function.draw_graph(graph.network)
            else:
                return_value = False
        elif chosen_strategy == 5:
            defense_DS_6(G_att_list, graph.sv, graph.ev, graph)
            return_value = True
        elif chosen_strategy == 6:
            defense_DS_7(P_fake_list, att_detect_list)
            return_value = True
        elif chosen_strategy == 7:
            defense_DS_8(graph.network, G_att_list, self.network, graph.using_honeynet,
                         graph.low_inter, graph.high_inter, graph.inter_per_node)
        else:
            pass
        return_value = True

    return return_value


# In[45]:


class defender_class:
    def __init__(self, game):
        if display: print("create defender")
        print('The scikit-learn version is {}.'.format(sklearn.__version__))
        self.network = copy.deepcopy(game.graph.network)  # defender's view
        self.strategy_number = game.strategy_number  # strategy size
        self.CKC_number = game.CKC_number
        self.CKC_list = []
        self.key_time = 1
        self.monit_time = 1
        self.dec = 0
        self.cost_limit = 3 # was 5
        self.strat_cost = def_strategy_cost(self.strategy_number)
        self.impact_record = np.ones(self.strategy_number)
        self.expected_impact = np.ones(self.strategy_number)
        #         self.strat_option = create_bundle(CKC_list, self.strategy_number, game.DD_using, self.strat_cost, self.cost_limit) # Table 4
        #         self.optional_bundle_list = defender_class_create_bundle(game.DD_using)
        self.optional_bundle_list = []  # spare bundles
        self.chosen_strategy_list = []  # selected bundle
        self.prob_believe_opponent = np.zeros(
            (game.CKC_number + 1, game.attacker_template.strategy_number))
        self.obs_oppo_strat_history = np.zeros(
            (game.CKC_number + 1, game.attacker_template.strategy_number))
        self.P_subgame = np.zeros(game.CKC_number + 1)  # Eq. 5: belief context
        self.subgrame_history = np.zeros(game.CKC_number + 1)
        self.deception_tech_used = False
        self.uncertain_scheme = game.uncertain_scheme_def
        self.scheme_name = game.scheme_name
        self.decision_scheme = game.decision_scheme
        if self.decision_scheme == 0:
            self.uncertainty = 1
        else:
            if self.uncertain_scheme:
                self.uncertainty = 1  # test, orignial 1
            else:
                self.uncertainty = 0


        self.DHEU = np.zeros(self.strategy_number)
        self.EU_C = None
        self.EU_CMS = None
        self.observed_strategy_count = np.zeros((self.CKC_number, self.strategy_number))
        self.observed_CKC_count = np.zeros(self.CKC_number)
        self.ML_window_size = 5
        self.s_j_window_record = np.zeros((self.ML_window_size, self.strategy_number))
        self.S_j = np.ones(self.strategy_number)/self.strategy_number
        self.use_bundle = game.use_bundle
        self.att_previous_strat = np.zeros(self.strategy_number)
        self.att_previous_CKC = np.zeros(self.CKC_number)


    create_bundle = defender_class_create_bundle
    choose_bundle = defender_class_choose_bundle
    execute_strategy = defender_class_execute_strategy


    def observe_opponent(self, attacker_list):
        # observe opponent action in one game
        for attacker in attacker_list:
            # identify player(attacker)
            if random.random() > self.uncertainty:
                # observe action
                if random.random() > self.uncertainty:
                    observed_action_id = attacker.chosen_strategy
                    self.att_previous_strat[attacker.chosen_strategy] += 1
                else:
                    # if unsuccessful to observe action, randomly guess one
                    observed_action_id = random.randrange(0,len(self.observed_strategy_count))

                # observe CKC
                if random.random() > self.uncertainty:
                    observed_CKC_id = attacker.CKC_position
                    self.att_previous_CKC[attacker.CKC_position] += 1
                else:
                    # if unsuccessful to observe CKC position, randomly guess one
                    observed_CKC_id = random.randrange(0,len(self.observed_CKC_count))

                self.observed_strategy_count[observed_CKC_id, observed_action_id] += 1    # for test
                self.observed_CKC_count[observed_CKC_id] += 1                         # for test



    def update_attribute(self, attacker_list):
        # key_time
        self.key_time += 1
        # monitor time
        if attacker_list:
            self.monit_time = np.sum([attacker.defender_monit_time for attacker in attacker_list])
        else:
            self.monit_time = 0
        # uncertainty
        # self.uncertainty = defender_uncertainty_update(att_detect, self.monit_time,
        #                                                self.uncertain_scheme, self.decision_scheme)
        uncertain_sum = 0
        counter = 0
        for attacker in attacker_list:
            uncertain_sum += attacker.defender_uncertainty
            counter += 1
        if uncertain_sum == 0:
            self.uncertainty = 0
        else:
            self.uncertainty = uncertain_sum/counter

    def reset_attribute(self, CKC_number):
        self.observed_strategy_count = np.zeros((self.CKC_number, self.strategy_number))    # add for test
        self.observed_CKC_count = np.zeros(self.CKC_number)     # add for teset
        self.key_time = 1
        self.monit_time = 1

        self.belief_context = [1 / (CKC_number + 1)] * (CKC_number + 1)
        #         self.obs_oppo_strat_history = np.zeros(
        #             (CKC_number + 1, self.strategy_number))
        self.dec = 0
        self.deception_tech_used = False
        if self.uncertain_scheme:
            self.uncertainty = 0  # test, orignial 1
        else:
            self.uncertainty = 0




    def decide_CKC_posi(self, att_CKC_list):
        # Selection Scheme. In 'random', defender randomly select bundle without consider CKC
        if self.decision_scheme == 0:  # for random selection scheme
            self.CKC_list = [0,1,2,3,4,5]
            return

        self.CKC_list = []
        g = self.uncertainty
        for att_CKC in att_CKC_list:
            if random.random() > g:
                self.CKC_list.append(att_CKC)

    def update_defense_impact(self, all_attack_impact, att_strat_list, strat_number):
        # if defender doesn't choose any strategy
        if not self.chosen_strategy_list:
            return
        # if attacker doesn't choose any strategy
        if not att_strat_list:
            return

        xi = 5
        # di = 0
        # for att_impact in all_attack_impact:
        #     di += math.exp(-1 * xi * att_impact)
        #     if att_impact != 0:
        #         print("the value")
        #         print(math.exp(-1 * xi * att_impact))

        strat_count = [att_strat_list.count(stra) for stra in np.arange(strat_number)]
        strat_prob = np.array(strat_count)/sum(strat_count)
        for strategy_id in self.chosen_strategy_list:
            self.impact_record[strategy_id] = sum([strat_prob[stra] * math.exp(-1 * xi * all_attack_impact[stra]) for stra in np.arange(strat_number)])
            # self.impact_record[strategy_id] = math.exp(-1 * xi * sum(all_attack_impact))  # old function


# In[ ]:


# In[ ]:


# In[ ]:
