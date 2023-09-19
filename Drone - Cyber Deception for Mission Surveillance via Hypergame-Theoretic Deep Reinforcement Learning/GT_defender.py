'''
Project     ：Drone-DRL-HT 
File        ：GT_defender.py
Author      ：Zelin Wan
Date        ：7/8/23
Description : This is the Game Theory Defender.Most of the code is the same as HT_defender.py except making the
uncertainty 0 !!!
'''

from HT_defender import HypergameTheoryDefender

class GameTheoryDefender(HypergameTheoryDefender):
    def __init__(self, env, **kwargs):
        HypergameTheoryDefender.__init__(self, env, **kwargs)
        self.uncertainty = 0.0

    def update_uncertainty(self):
        self.uncertainty = 0.0




