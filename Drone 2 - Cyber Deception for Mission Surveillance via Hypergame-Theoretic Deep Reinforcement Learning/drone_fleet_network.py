'''
Project     ：Drone-DRL-HT 
File        ：drone_fleet_network.py
Author      ：Zelin Wan
Date        ：2/4/23
Description : Drone fleet network based on NetworkX package
'''

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from model_HD import Honey_Drone
from model_MD import Mission_Drone


def create_drone_fleet(MD_num=8, HD_num=2):
    '''
    ID start from HD, then MD
    :param MD_num:
    :param HD_num:
    :return:
    '''
    fleet_G = nx.complete_graph(MD_num+HD_num)

    # assign honey drone to nodes
    for HD_id in range(HD_num):
        fleet_G.nodes[HD_id]['class'] = Honey_Drone(HD_id)
    # assign mission drone to nodes
    for MD_id in range(HD_num, HD_num+MD_num):
        fleet_G.nodes[MD_id]['class'] = Mission_Drone(MD_id)

    for node_id in range(fleet_G.number_of_nodes()):
        fleet_G.nodes[node_id]['pos'] = np.zeros(2)

    return fleet_G


fleet_G = create_drone_fleet()
# for node_id, node in fleet_G.nodes(data=True):
#     print(node_id, node)
# MD_set = {node['class'].ID: node['class'] for node in fleet_G.nodes() if node['class'].type is 'MD'}
print(fleet_G.nodes())
# print(fleet_G.nodes[0])
# pos_dict = nx.get_node_attributes(fleet_G, 'pos')
# nx.draw(fleet_G, pos_dict)
# plt.show()









