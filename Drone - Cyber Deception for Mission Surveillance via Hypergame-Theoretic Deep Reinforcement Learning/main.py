# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from Gym_Defender_and_Attacker import HyperGameSim
import networkx as nx
import matplotlib.pyplot as plt


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    env = HyperGameSim()
    print(env.action_space['att'].n)
    obs_old = env.reset()

    episodes = 1
    for epi in range(episodes):
        done = False
        while not done:
            the_obs_, the_reward, done, info = env.step()
            obs_old = the_obs_
            print(env.system.scan_cell_map)
            env.system.draw_MD_Position()







