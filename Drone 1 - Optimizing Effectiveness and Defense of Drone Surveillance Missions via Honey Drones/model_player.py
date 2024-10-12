import numpy as np
import random

class player_model:
    def __init__(self, system):
        self.print = False
        self.system = system  # save the pointer to system class
        self.xyz = [0, 0, 0]  # player's location
        self.strategy = 9
        self.strategy_index_start = 0  # strategy index range. in paper, strategy start from 1, but in coding, strategy index start from 0 for convinience
        self.strategy_index_end = 9  # strategy index range
        self.number_of_strategy = 10    # total number of strategy

    def update_location(self, x_axis, y_axis, z_axis):
        self.xyz = (x_axis, y_axis, z_axis)

    def select_strategy(self, new_strategy):
        self.strategy = int(new_strategy)

    def action(self):
        pass