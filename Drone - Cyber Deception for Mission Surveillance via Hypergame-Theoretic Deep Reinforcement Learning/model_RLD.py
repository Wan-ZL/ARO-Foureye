'''
Project     ：Drone-DRL-HT 
File        ：model_RLD.py
Author      ：Zelin Wan
Date        ：2/4/23
Description : 
'''

from model_Drone import Drone
import numpy as np

class RLD_Drone(Drone):
    def __init__(self, ID):
        Drone.__init__(self, ID)
        self.type = "RLD"
        self.xyz = np.zeros(3)
        # self.signal = -8 # TODO: this is a test