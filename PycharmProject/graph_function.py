#!/usr/bin/env python
# coding: utf-8

# In[48]:


from main import display
from networkx import nx
from attacker_function import *
from defender_function import *
import time
import random
import math
import copy
import matplotlib.pyplot as plt



# In[47]:


# set ID in attribute
def set_id(G):
    if not nx.get_node_attributes(G, "id"):
        nx.set_node_attributes(G, 0, "id")
    for n in G.nodes():
        G.nodes[n]["id"] = n


# In[31]:


# set node type
# 0 means IoT devices, 1 means Web Server, 2 means Dataset, 3 means Honeypot
def set_type(G, N_ws, N_db):
    if "type" not in G.nodes[0]:
        nx.set_node_attributes(G, 0, "type")
    for n in range(N_ws):
        G.nodes[n]["type"] = 1
    for n in range(N_db, N_ws + N_db):
        G.nodes[n]["type"] = 2



def set_type_for_HP(H_G, low_inter, high_inter):
    if "type" not in H_G.nodes['LI0']:
        nx.set_node_attributes(H_G, 0, "type")
    for n in ["LI" + str(n) for n in range(low_inter)]:
        H_G.nodes[n]["type"] = 3
    for n in ["HI" + str(n) for n in range(high_inter)]:
        H_G.nodes[n]["type"] = 3
 


# In[ ]:


def adjacent_node(G, node_id):
    return_list = []
    adjacent_list = [i for i in G[node_id]]
    if G.nodes[node_id]["type"] == 3: # if node is honeynet
        for i in adjacent_list:
            if G.nodes[i]["type"] == 3: # only consider honeypot as neighbor
                return_list.append(i)
    else:
        return_list = adjacent_list
    
    return return_list
    


# In[ ]:


def number_of_edge_without_honeypot(G):
    temp_G = copy.deepcopy(G)
    all_nodes = copy.deepcopy(temp_G.nodes(data=False))
    for node_id in all_nodes:
        if temp_G.nodes[node_id]["type"] == 3:
            temp_G.remove_node(node_id)
    return temp_G.number_of_edges()


# In[32]:


# set honeypot attribute
# 0 means not a honeypot, 1 means low interaction honeypot, 2 means high interaction honeypot
def set_HP_attribute(G):
    if not nx.get_node_attributes(G, "honeypot"):
        nx.set_node_attributes(G, 0, "honeypot")
    for n in G.nodes():
        if G.nodes[n]["type"] == 3:
            if n[0] == 'L':
                G.nodes[n]["honeypot"] = 1
            else:
                G.nodes[n]["honeypot"] = 2


# In[ ]:


def ids_without_honeypot(G):
    all_nodes = G.nodes(data="honeypot")
    return_list = [index[0] for index in all_nodes if index[1]==0]
    return return_list


# In[33]:


# update time-based encryption vulnerability
def update_en_vul(G, ev, ev_lambda, T_rekey):
    T_rekey += 1
    for n in G.nodes():
        for index in range(ev):
            G.nodes[n]["encryption vulnerability"][index] = G.nodes[n][
                "original_encryption_vulnerability"][index] * math.exp(
                    -ev_lambda / T_rekey)


# In[ ]:


# update vulnerability_i value
# Call this function to update vulnerability_i calculation
def update_vul(G):
    if not nx.get_node_attributes(G, "vulnerability"):
        nx.set_node_attributes(G, 0, "vulnerability")
    if not nx.get_node_attributes(G, "normalized_vulnerability"):
        nx.set_node_attributes(G, 0, "normalized_vulnerability")
    for n in G.nodes():
        G.nodes[n]["vulnerability"] = (
            sum(G.nodes[n]["software vulnerability"]) +
            sum(G.nodes[n]["encryption vulnerability"]) +
            sum(G.nodes[n]["unknown vulnerability"])) / (
                len(G.nodes[n]["software vulnerability"]) +
                len(G.nodes[n]["encryption vulnerability"]) +
                len(G.nodes[n]["unknown vulnerability"]))
        G.nodes[n][
            "normalized_vulnerability"] = G.nodes[n]["vulnerability"] / 10
        
        if type(G.nodes[n]["normalized_vulnerability"]) == list:
            if len(G.nodes[n]["normalized_vulnerability"])>1:
                print("in graph")
                print(G.nodes[n]["normalized_vulnerability"])

    return G


# In[35]:


def set_security_vulnerability(G, sv, ev, uv, ev_lambda, T_rekey,
                               web_data_upper_vul, Iot_upper_vul):
    # set security vulnerability
    if not nx.get_node_attributes(G, "software vulnerability"):
        nx.set_node_attributes(G, [0] * sv, "software vulnerability")
    if not nx.get_node_attributes(G, "original_encryption_vulnerability"):
        nx.set_node_attributes(G, [0] * ev,
                               "original_encryption_vulnerability")
    if not nx.get_node_attributes(G, "encryption vulnerability"):
        nx.set_node_attributes(G, [0] * ev, "encryption vulnerability")
    if not nx.get_node_attributes(G, "unknown vulnerability"):
        nx.set_node_attributes(G, [0] * uv, "unknown vulnerability")

    # add three types vulnerability value based on essay TABLE 1
    for n in G.nodes():
        if G.nodes[n]["type"] == 0:  # if IoT
            G.nodes[n]["software vulnerability"] = [
                #                 random.randint(1, 5) for iter in range(sv)
                random.randint(1, Iot_upper_vul) for iter in range(sv)
            ]
            G.nodes[n]["original_encryption_vulnerability"] = [
                random.randint(5, 10) for iter in range(ev)
            ]
            G.nodes[n]["unknown vulnerability"] = [
                random.randint(0, 10) for iter in range(uv)
            ]
        elif G.nodes[n]["type"] == 1:  # if Web Server
            G.nodes[n]["software vulnerability"] = [
                #                 random.randint(3, 7) for iter in range(sv)
                random.randint(1, web_data_upper_vul) for iter in range(sv)
            ]
            G.nodes[n]["original_encryption_vulnerability"] = [
                random.randint(1, 3) for iter in range(ev)
            ]
            G.nodes[n]["unknown vulnerability"] = [
                random.randint(0, 10) for iter in range(uv)
            ]
        elif G.nodes[n]["type"] == 2:  # if Dataset
            G.nodes[n]["software vulnerability"] = [
                #                 random.randint(3, 7) for iter in range(sv)
                random.randint(1, web_data_upper_vul) for iter in range(sv)
            ]
            G.nodes[n]["original_encryption_vulnerability"] = [
                random.randint(1, 3) for iter in range(ev)
            ]
            G.nodes[n]["unknown vulnerability"] = [
                random.randint(0, 10) for iter in range(uv)
            ]
        else:  # if Honeypot
            G.nodes[n]["software vulnerability"] = [
                random.randint(7, 10) for iter in range(sv)
            ]
            G.nodes[n]["original_encryption_vulnerability"] = [
                random.randint(9, 10) for iter in range(ev)
            ]
            G.nodes[n]["unknown vulnerability"] = [
                random.randint(0, 10) for iter in range(uv)
            ]

    # update encryption vulnerability
    update_en_vul(G, ev, ev_lambda, T_rekey)
    # update overall vulnerability
    update_vul(G)


# In[36]:


def set_importance(G):
    if not nx.get_node_attributes(G, "importance"):
        nx.set_node_attributes(G, 0, "importance")
    for n in G.nodes():
        if G.nodes[n]["type"] == 0:
            G.nodes[n]["importance"] = random.randint(1, 5)
        elif G.nodes[n]["type"] == 1:
            G.nodes[n]["importance"] = random.randint(8, 10)
        elif G.nodes[n]["type"] == 2:
            G.nodes[n]["importance"] = random.randint(8, 10)


# In[37]:


# Mobility
def set_mobility(G):
    if not nx.get_node_attributes(G, "mobility"):
        nx.set_node_attributes(G, 0, "mobility")
    for n in G.nodes():
        if G.nodes[n]["type"] == 0:  # only for IoT devices
            G.nodes[n]["mobility"] = 0.1  #random.uniform(0,0.5)


# In[38]:


# Compromised Status
# For devices: False means not compromised, True means compromised.
# For honeypots: False means not visited by attacker, True means visited by attacker


def set_compromised_status(G):
    if not nx.get_node_attributes(G, "compromised_status"):
        nx.set_node_attributes(G, False, "compromised_status")


# In[ ]:


# Evicted mark


def set_evicted_mark(G):
    if not nx.get_node_attributes(G, "evicted_mark"):
        nx.set_node_attributes(G, False, "evicted_mark")


# In[43]:


# reachability
# isolated node have betweenness value 0.0
def set_reachability(G):
    if not nx.get_node_attributes(G, "reachability"):
        nx.set_node_attributes(G, 0, "reachability")
    reachability = nx.betweenness_centrality(G)
    # reachability = nx.degree_centrality(G)
    for n in G.nodes():
        G.nodes[n]["reachability"] = reachability[n]


# In[45]:


# Criticality
def update_criticality(G):

    set_reachability(G)

    if not nx.get_node_attributes(G, "reachability"):
        nx.set_node_attributes(G, 0, "reachability")
    for n in G.nodes():
        G.nodes[n]["criticality"] = G.nodes[n]["importance"] * G.nodes[n][
            "reachability"]


# In[41]:



def graph_attrbute(G, sv, ev, uv, ev_lambda, T_rekey, web_data_upper_vul, Iot_upper_vul):

    # set id
    set_id(G)
        
    # set honeypot
    set_HP_attribute(G)


    # set vulnearbility
    set_security_vulnerability(G, sv, ev, uv, ev_lambda, T_rekey, web_data_upper_vul, Iot_upper_vul)
   

    # set importance
    set_importance(G)


    # set mobility
    set_mobility(G)

    # set compromised status
    set_compromised_status(G)
    
    # set evicted status
    set_evicted_mark(G)

    # update criticality
    update_criticality(G)
    
    # Recheck: zero result for honeypot


# In[46]:


# print node data
def show_all_nodes(G):
    if G is None:
        print("G is None")
        return
    
    for n in G.nodes():
        if display: print(n)
        if G.nodes[n]["compromised_status"]:
            print("\x1b[6;73;41m", G.nodes[n], "\x1b[0m")
        else:
            print(G.nodes[n])


# In[4]:


# draw with color
def draw_graph(G):
    if G is None:
        if display: print("Failed Draw Graph")
        return
    
    file_name = str(round(time.time()*10))
    
    plt.figure()
#     groups = set(nx.get_node_attributes(G, 'honeypot').values())
#     groups.add(3) # for compromised node
#     mapping = dict(zip(sorted(groups), count()))
#     colors = [mapping[3 if G.nodes[n]['compromised_status'] else G.nodes[n]['honeypot']] for n in G.nodes()]
    colors = []
    for n in G.nodes():
        if G.nodes[n]["compromised_status"]:
            colors.append('#FF0000')
        else:
            if G.nodes[n]["honeypot"] == 0:
                colors.append('#9932CC')
            elif G.nodes[n]["honeypot"] == 1:
                colors.append('#008000')
            else:
                colors.append('#CCCC00')

#     compro_dict = dict((k, G.nodes[k]["honeypot"]) for k in G.nodes())
    
    options = {
        "pos": nx.circular_layout(G),
        "node_color": colors,
        "node_size": 20,
        "arrowsize": 3,
        "line_color": "grey",
        "linewidths": 0,
        "width": 0.1,
        "with_labels": True,
        "font_size": 3,
        "font_color": 'w',
#         "labels": rounded_vul,
    }
    nx.draw(G,  **options)
    if display: print(G)
    plt.savefig("graph/graph"+file_name+".png", dpi=1000)


# In[ ]:


def is_node_evicted(G, target_id):
    return G.nodes[target_id]["evicted_mark"]


# In[ ]:


# is all compromised node evicted
def is_all_evicted(G_real, compromised_nodes):
    for index in compromised_nodes:
        if G_real.has_node(index):
            if not is_node_evicted(G_real, index):
                return False
    
    if display: print(f"all evicted {is_all_evicted}")  
    return True
    


# In[ ]:


def number_of_evicted_node(G):
    all_evicted_mark = list(nx.get_node_attributes(G, "evicted_mark").values())
    return sum(all_evicted_mark)


# In[ ]:


def evict_a_node(remove_id, G_real, G_def, G_att):
    
    node_neighbor = list(G_def.neighbors(remove_id))
    
    # remove edge to adjacent nodes
    for neighbor_index in node_neighbor:
        if G_real.has_edge(remove_id,neighbor_index): G_real.remove_edge(remove_id,neighbor_index)
        if G_def.has_edge(remove_id,neighbor_index): G_def.remove_edge(remove_id,neighbor_index)
        if G_att.has_edge(remove_id,neighbor_index): G_att.remove_edge(remove_id,neighbor_index)
    
    # change evict mark
    G_real.nodes[remove_id]["evicted_mark"] = True
    G_def.nodes[remove_id]["evicted_mark"] = True
    G_att.nodes[remove_id]["evicted_mark"] = True
        
    # update criticality
    update_criticality(G_real)
    update_criticality(G_def)
    update_criticality(G_att)


# In[3]:


def evict_all_compromised(G_real, G_att, G_def):
    for index in G_real.nodes:
            if G_real.nodes[index]["compromised_status"]:
                evict_a_node(index, G_real, G_def, G_att)


# In[ ]:


def is_connect_DataServer(target_id, G):
    nodes_in_same_component = list(nx.node_connected_component(G, target_id))
    if len(nodes_in_same_component) == 1:
        return False
    for node_id in nodes_in_same_component:
        if G.nodes[node_id]["type"] == 1 or G.nodes[node_id]["type"] == 2:
            return True


# In[ ]:


def connect_to_WS_DB_component(target_id, G_real, G_def, G_att):
    node_list = list(G_real.nodes(data=False))
    random.shuffle(node_list)
    for node_id in node_list:
        # find a WS or DB
        if G_real.nodes[node_id]["type"] == 1 or G_real.nodes[node_id]["type"] == 2:
            # get the component of WS or DB node
            nodes_in_component = list(nx.node_connected_component(G_real, node_id))
            selected_id = random.choice(nodes_in_component)
            G_real.add_edge(target_id, selected_id)
            G_def.add_edge(target_id, selected_id)
            G_att.add_edge(target_id, selected_id)
            return
            


# In[1]:


# connect a non-evicted node to Webserver of Database if it's not.
def reconnect_a_node(target_id, G_real, G_def, G_att, connect_prob):
    if not is_node_evicted(G_real, target_id):
        if not is_connect_DataServer(target_id, G_real):
            new_edge_number = int(connect_prob * (G_real.number_of_nodes()-number_of_evicted_node(G_real)))
            while(new_edge_number >= 0):
                connect_to_WS_DB_component(target_id, G_real, G_def, G_att)
                new_edge_number -= 1
    return
                
                
#             node_list = list(G_real.nodes(data=False))
#             random.shuffle(node_list)
#             for node_id in node_list:
#                 if G_real.nodes[node_id]["type"] == 1 or G_real.nodes[node_id]["type"] == 2:
#                     G_real.add_edge(target_id, node_id)
#                     G_def.add_edge(target_id, node_id)
#                     G_att.add_edge(target_id, node_id)
#                     return True
#     return False


# In[ ]:


def node_reconnect(G_real, G_att, G_def, connect_prob):
    for node_id in G_real.nodes(data=False):
        reconnect_a_node(node_id, G_real, G_def, G_att, connect_prob)


# In[ ]:


# Check System Failure

def is_system_fail(graph, reason_box, SF_thres_1, SF_thres_2):
    # SF_thres_1 = 1/3 # may try 1/5  # A threshold for SF (Rho_1)
    # SF_thres_2 = 1/2 # Rho_2
    
    G_real = graph.network
    
    # Rho_2
    counter = 0
    total_node_number = 0
    for node_id in G_real.nodes():
        if G_real.nodes[node_id]["type"] != 3:   # ignore honeypot
            total_node_number += 1
            if not G_real.nodes[node_id]["evicted_mark"]:   # if not evicted
                counter += 1

    if SF_thres_2 >= counter/total_node_number:
        reason_box[0] = 2
        return True

    
    
    # Rho_1
    top_total = 0;
    bottom_total = 0;
    for n in G_real.nodes():
        top_total += G_real.nodes[n]["compromised_status"] * G_real.nodes[n]["importance"]
        bottom_total += G_real.nodes[n]["importance"]
    if bottom_total == 0.0:
        return False
    if top_total/bottom_total >= SF_thres_1:
        reason_box[0] = 1
        return True
    else:
        return False
    
    


# In[ ]:


def rewire_network(G_real, G_att, G_def, rewire_prob):
#     # if all node evicted, do nothing (It's OK to remove it)
#     all_evict_mark = list(nx.get_node_attributes(G_real, "evicted_mark").values())
# #     print(f"sum(all_evict_mark) {sum(all_evict_mark)}")
# #     print(f"len(all_evict_mark) {len(all_evict_mark)}")
#     if sum(all_evict_mark) >= len(all_evict_mark)-3:
#         print("rewire_network Fail")
#         return
    
    for index in G_real.nodes(data=False):
        discon_node = None
        if not is_node_evicted(G_real, index):    # don't rewire evicted node
            if random.random() < rewire_prob:
                # select node to disconnect
                adj_list = adjacent_node(G_real, index)
                if adj_list:      # if have neighbor
                    discon_node = random.choice(adj_list)
        
        if discon_node is not None:
            # select node to reconnect
            recon_node = None
            all_node = list(G_real.nodes(data=False))
            if index in all_node: all_node.remove(index)  # avoid connect to itself
            while recon_node is None:
                selected_node = random.choice(all_node)
                if not is_node_evicted(G_real, selected_node):   # # don't rewire evicted node
                    recon_node = selected_node
            
            # disconnect
            if G_real.has_edge(index, discon_node): G_real.remove_edge(index, discon_node)
            if G_att.has_edge(index, discon_node): G_att.remove_edge(index, discon_node)
            if G_def.has_edge(index, discon_node): G_def.remove_edge(index, discon_node)
            # reconnect
            if display: print(f"Rewire: remove {[index, discon_node]}, connect {[index, recon_node]}")
            if not G_real.has_edge(index, recon_node): G_real.add_edge(index, recon_node)
            if not G_att.has_edge(index, recon_node): G_att.add_edge(index, recon_node)
            if not G_def.has_edge(index, recon_node): G_def.add_edge(index, recon_node)
            
            
            
#         if random.random() < rewire_prob:
#             # disconnect a node
#             adj_list = adjacent_node(G_real, index)
#             if adj_list:  # if have neighbor
#                 discon_node = random.choice(adj_list)
#                 if G_real.has_edge(index, discon_node): G_real.remove_edge(index, discon_node)
#                 if G_att.has_edge(index, discon_node): G_att.remove_edge(index, discon_node)
#                 if G_def.has_edge(index, discon_node): G_def.remove_edge(index, discon_node)
#                 # reconnect a node
#                 all_node = list(G_real.nodes(data=False))
#                 if index in all_node: all_node.remove(index) # avoid connect to itself
#                 recon_node = random.choice(all_node)
#                 if display: print(f"Rewire: remove {[index, discon_node]}, connect {[index, recon_node]}")
#                 if not G_real.has_edge(index, recon_node): G_real.add_edge(index, recon_node)
#                 if not G_att.has_edge(index, recon_node): G_att.add_edge(index, recon_node)
#                 if not G_def.has_edge(index, recon_node): G_def.add_edge(index, recon_node)
                
            
            
        


# In[ ]:


def graph_class_create_graph(self):
    self.network = nx.erdos_renyi_graph(self.node_number,
                                        self.connect_prob)  # undirected graph
    while not nx.is_connected(self.network):
        if display: print("rebuild")
        self.network = nx.erdos_renyi_graph(
            self.node_number, self.connect_prob)  # rebuild undirected graph

    set_type(self.network, self.N_ws, self.N_db)
    graph_attrbute(self.network, self.sv, self.ev, self.uv, self.ev_lambda,
                   self.T_rekey, self.web_data_upper_vul, self.Iot_upper_vul)


#     plt.figure()
#     nx.draw(self.network, with_labels=True)
#     return self.network


# In[ ]:


def graph_class_new_honeypot(self):
    self.honey_net = nx.complete_graph(
        self.low_inter + self.high_inter)  # new graph for honeypot
    mapping = {}
    for n in range(self.low_inter):
        mapping[n] = "LI" + str(n)
    for n in range(self.high_inter):
        mapping[n + self.low_inter] = "HI" + str(n)
    self.honey_net = nx.relabel_nodes(self.honey_net, mapping)
    if display: print("honeypot graph")
    #         self.honey_net = self.honey_net.to_directed()

    set_type_for_HP(self.honey_net, self.low_inter, self.high_inter)
    graph_attrbute(self.honey_net, self.sv, self.ev, self.uv, self.ev_lambda,
                   self.T_rekey, self.web_data_upper_vul, self.Iot_upper_vul)


#         plt.figure()
#         nx.draw(self.honey_net, with_labels=True)


# In[ ]:


def clean_honeynet(G_real, G_att, G_def):
        all_node = copy.deepcopy(G_real.nodes())
        counter = 0
        for node_id in all_node:
            if G_real.nodes[node_id]["type"] == 3:
                G_real.remove_node(node_id)
        all_node = copy.deepcopy(G_att.nodes())
        for node_id in all_node:
            if G_att.nodes[node_id]["type"] == 3:
                G_att.remove_node(node_id)
        all_node = copy.deepcopy(G_def.nodes())
        for node_id in all_node:
            if G_def.nodes[node_id]["type"] == 3:
                G_def.remove_node(node_id)


# In[ ]:


class graph_class:
    def __init__(self, web_data_upper_vul=7, Iot_upper_vul=5):
        self.network = None
        self.honey_net = None
        self.using_honeynet = False
        self.network_size_factor = 5 #5
        self.node_number = 100*self.network_size_factor#100  # number of nodes
        self.connect_prob = 0.05  # connection probability
        self.SF_thres = 0.3  # A threshold for SF
        self.low_inter = 10*self.network_size_factor #10  # number of low interaction honeypots
        self.high_inter = 5*self.network_size_factor #5  # number of high interaction honeypots
        self.inter_per_node = 3 # one honeypot connect to 3 nodes
        self.N_ws = 5*self.network_size_factor #5  # number of Web servers
        self.N_db = 5*self.network_size_factor #5  # number of databases
        self.N_iot = self.node_number - self.N_ws - self.N_db  # number of IoT nodes
        self.ev = 5  # encryption vulnerability
        self.sv = 5  # software vulnerability
        self.uv = 1  # unknown vulnerability
        self.ev_lambda = 1 # λ for normalize encryption vulnerability
        self.T_rekey = 1 # rekey time for encryption vulnerability
        self.web_data_upper_vul = web_data_upper_vul
        self.Iot_upper_vul = Iot_upper_vul
        
        if display: print("create graph")
        self.create_graph()
        
    def T_rekey_reset(self):
        self.T_rekey = 1
        
    create_graph = graph_class_create_graph
        
    new_honeypot = graph_class_new_honeypot
    
    def update_graph(self, G_def, G_att):
        update_criticality(self.network)
        update_criticality(G_def)
        update_criticality(G_att)
        update_vul(self.network)
        update_vul(G_def)
        update_vul(G_att)
        update_en_vul(self.network, self.ev, self.ev_lambda, self.T_rekey)
        update_en_vul(G_def, self.ev, self.ev_lambda, self.T_rekey)
        update_en_vul(G_att, self.ev, self.ev_lambda, self.T_rekey)
        
                
        
        
    
    
        

