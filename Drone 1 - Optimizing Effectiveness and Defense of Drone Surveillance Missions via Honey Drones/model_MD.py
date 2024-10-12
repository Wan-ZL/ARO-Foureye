import numpy as np
from model_Drone import Drone

class Mission_Drone(Drone):
    def __init__(self, ID, update_freq):
        Drone.__init__(self, ID, update_freq)
        self.type = "MD"
        # self.battery_max = 100.0
        # self.battery = self.battery_max  # battery level
        self.charging_rate = 0.02
        self.consume_rate = self.E_P + self.E_C + self.E_R
        self.surveyed_time = 0
        self.surveyed_cell = np.array([])     # cell ID goes here





        