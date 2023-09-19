'''
Project     ：Drone-DRL-HT 
File        ：model_Drone.py
Author      ：Zelin Wan
Date        ：2/4/23
Description : 
'''

import numpy as np

class Drone:
    def __init__(self, ID):
        self.print_debug = False
        self.ID = ID
        self.type = 'Drone'
        self.died = False  # drone died when battery goes to zero
        # self.charging = True
        self.xyz_destination = np.zeros(3)  # Droen location (location of destination)
        self.xyz_temp = self.xyz_destination.copy()  # intermediate location to destination
        self.speed_per_frame_max = 0.08     # this value obtained from experiment that drone doesn't crash for a 150 meter fly in one round
        self.crashed = False             #
        self.in_GCS = True
        self.battery_max = 750 # 100000.0         # battery level (B078ZZPZ5Z battery on Amazon)
        self.battery = self.battery_max
        self.E_P = 250.0/7/60 # 0.001   # 60 is 60 seconds in one minute. 250.0/7 is estimated drone platform energy rate (fly time 7 minutes)
        self.E_C = 0.7
        self.E_R = 0.1
        self.consume_rate = self.E_P + self.E_C + self.E_R
        self.accumulated_consumption = 0        # this will show the total energy consumption in one episode
        self.max_signal = 20                # unit: dBm (fixed)
        self.signal = self.max_signal       # unit: dBm
        self.signal_radius = 1000           # unit meter
        self.been_attack_record = (0,0)     # the first element is # of success, the second element is # of failure.
        self.neighbor_table = {}  # key: drone ID, value: True means connected to this neighbor, False means not connected.
        self.connect_RLD = True # True means connected to RLD, False means disconnected from RLD.
        self.visited = False    # design for DFS search
        self.memory_full = 0  # design for container drone (CD)

    def battery_update(self):       # consume energy or charging (True means drone is ready (recalculate trajectory))
        if self.crashed:        # ignore crashed drone
            return False

        if self.in_GCS:
            if self.battery < self.battery_max: # drone in GCS with not full battery will charge
                self.battery += self.charging_rate
                # send signal if battery full
                if self.battery >= self.battery_max:
                    return True
        else:                                   # any drone not in GCS consume energy
            self.battery -= self.consume_rate
            self.accumulated_consumption += self.consume_rate
        return False

    def consume_rate_update(self, DS_j):
        self.consume_rate = self.E_P + self.E_C + self.E_R * (DS_j/10)



    # check if a drone should change from normal condition to crashed condition
    def new_crash(self, xyz_current):
        if self.crashed:
            return False
        if not self.in_GCS:     # drone in GCS always safe
            if self.battery <= 0:
                if self.print_debug: print("\n====Drone crashed by zero battery====, ID:", self.ID, self.type, "\n")
                self.crashed = True
                return True
            if xyz_current[2] < 0.1:
                self.crashed = True
                if self.print_debug: print("\n====Drone crashed by zero height====, ID:", self.ID, self.type, xyz_current, "\n")
                if self.print_debug: print(self)
                self.crashed = True
                return True
        return False

    def assign_destination(self, xyz_destination):
        if self.battery < self.low_battery_thres:   # low battery go charging
            self.go_charging()
        elif not self.crashed:
            self.xyz_destination = xyz_destination

    # assigning x y but keep z
    def assign_destination_xy(self, xy_destination):
        if self.battery < self.low_battery_thres:  # low battery go charging
            self.go_charging()
        elif not self.crashed:
            self.xyz_destination[:2] = xy_destination

    # this value is based on the consumption in one round
    # @property make this function as variable so '()' is not required when calling it (same as Getter in Java)
    @property
    def low_battery_thres(self):
        return (self.E_P + self.E_C + self.E_R) + 0.1

    def __repr__(self):
        return str(vars(self))


    def condition_check(self):
        if not self.crashed:
            if np.array_equal(self.xyz_destination[:2], np.zeros(2)):
                self.in_GCS = True
            else:
                self.in_GCS = False

    def go_charging(self):
        self.xyz_destination[:2] = np.zeros(2)
