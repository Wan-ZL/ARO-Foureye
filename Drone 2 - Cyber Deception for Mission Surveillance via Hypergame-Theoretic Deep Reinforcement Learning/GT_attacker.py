'''
Project     ：Drone-DRL-HT 
File        ：GT_attacker.py
Author      ：Zelin Wan
Date        ：7/8/23
Description : This is the Game Theory Attacker. Most of the code is the same as HT_attacker.py except making the
uncertainty 0 !!!
'''

from HT_attacker import HypergameTheoryAttacker

class GameTheoryAttacker(HypergameTheoryAttacker):
    def __init__(self, env, **kwargs):
        HypergameTheoryAttacker.__init__(self, env, **kwargs)
        self.uncertainty = 0.0

    def update_uncertainty(self):
        self.uncertainty = 0.0