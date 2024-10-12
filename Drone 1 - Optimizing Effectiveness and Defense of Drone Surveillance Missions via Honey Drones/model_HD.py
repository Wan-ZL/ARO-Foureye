import numpy as np
from model_Drone import Drone

class Honey_Drone(Drone):
    def __init__(self, ID, update_freq):
        Drone.__init__(self, ID, update_freq)
        self.type = "HD"
        # self.battery_max = 100.0
        # self.battery = self.battery_max  # battery level
        self.charging_rate = 0.02
        self.consume_rate = self.E_P + self.E_R
        self.protecting = np.array([])        # the MD (ID) protecting now

    def consume_rate_update(self, DS_j):
        self.consume_rate = self.E_P + self.E_R * (DS_j/10)