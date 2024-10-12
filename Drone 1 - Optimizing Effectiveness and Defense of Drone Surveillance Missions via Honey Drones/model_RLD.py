'''
Project     ：gym-drones 
File        ：model_RLD.py
Author      ：Zelin Wan
Date        ：10/11/22
Description : 
'''

from model_Drone import Drone
import numpy as np

class RLD_Drone(Drone):
    def __init__(self, ID, update_freq):
        Drone.__init__(self, ID, update_freq)
        self.type = "RLD"
        self.xyz = np.zeros(3)
        self.signal = -8