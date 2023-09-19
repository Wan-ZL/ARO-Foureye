'''
Project     ：Drone-DRL-HT 
File        ：HT_interaction.py
Author      ：Zelin Wan
Date        ：3/27/23
Description : Two Hypergame Theory Agents interact with each other
'''
import random
import pickle
import os
import time
import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np

from HT_attacker import HypergameTheoryAttacker
from HT_defender import HypergameTheoryDefender
from Gym_Defender_and_Attacker import HyperGameSim
from torch.utils.tensorboard import SummaryWriter

def train_for_HEU_distribution(episodes=100, ID=0):
    '''
    Train the HEU for the defender and attacker
    :param episodes:
    :return:
    '''

    # initialize the tensorboard writer for each run (for experiment data collection
    current_time = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    file_name = "each_run-Time_" + current_time + "-ID_" + str(ID)

    enable_experi_writer = True  # True to enable tensorboard writer for environment results, False to disable.
    if enable_experi_writer:
        file_path = "data/HEU/experiment_results/"
        os.makedirs(file_path, exist_ok=True)
        experi_writer = SummaryWriter(log_dir=file_path + file_name)
    else:
        experi_writer = None

    # initialize the tensorboard writer for each agent
    enable_agent_writer = True  # True to enable tensorboard writer for agent results, False to disable.
    if enable_agent_writer:
        file_path = "data/HEU/agent_att/"
        os.makedirs(file_path, exist_ok=True)
        att_writer = SummaryWriter(log_dir=file_path + file_name)
        file_path = "data/HEU/agent_def/"
        os.makedirs(file_path, exist_ok=True)
        def_writer = SummaryWriter(log_dir=file_path + file_name)
    else:
        att_writer = None
        def_writer = None

    # initialize the environment
    env = HyperGameSim()

    # initialize the attacker and defender
    attacker = HypergameTheoryAttacker(env, writer=att_writer)
    defender = HypergameTheoryDefender(env, writer=def_writer)
    # start the interaction
    for episode in range(episodes):
        print("start episode: ", episode, " in run: ", ID)

        # initialize the state
        state = env.reset()
        state_def = state['def']
        state_att = state['att']
        done = False
        info = {}
        step = 0
        # start the episode
        while not done:
            step += 1

            # attacker takes an action
            action_att = attacker.act()
            # action_att = random.randint(0, 9)

            # defender takes an action
            action_def = defender.act()
            # action_def = random.randint(0, 9)

            # environment takes an step
            next_state, reward, done, info = env.step(action_def=action_def, action_att=action_att)
            next_state_def = next_state['def']
            next_state_att = next_state['att']
            reward_def = reward['def']
            reward_att = reward['att']

            # print("def strategy (env)", defender.env.defender.strategy)
            # print("def strategy", defender.self_strategy)
            # print("def's att strategy", defender.oppo_strategy)
            # print("action counter", defender.action_counter)
            # print("def cost", defender.cost)
            # print("def oppo cost", defender.oppo_cost)
            # print("def utility", defender.utility)
            # print("=====================================")

            # Players observe opponent and environment
            attacker.observe(action_def)
            defender.observe(action_att)

            # update the state
            state_def = next_state_def
            state_att = next_state_att



        if experi_writer is not None:
            # write the result to tensorboard
            for key in info.keys():
                experi_writer.add_scalar(key, info[key], episode)
            experi_writer.add_scalar("att_succ_ratio", attacker.att_succ_ratio, episode)

        if att_writer is not None:
            att_writer.add_scalar("steps in each episode", step, episode)
        if def_writer is not None:
            def_writer.add_scalar("steps in each episode", step, episode)



        # show the result
        # print("Episode: ", episode)
        # print("attacker HEU_prob: ", attacker.HEU_prob)
        # print("defender HEU_prob: ", defender.HEU_prob)
        # print("attacker uncertainty: ", attacker.uncertainty)
        # print("defender uncertainty: ", defender.uncertainty)
        # print("=====================================")

    # close the tensorboard writer
    if experi_writer is not None:
        experi_writer.flush()
        experi_writer.close()
    if att_writer is not None:
        att_writer.flush()
        att_writer.close()
    if def_writer is not None:
        def_writer.flush()
        def_writer.close()

    # return the Hypergame Expectation Utility for both attacker and defender.
    # return the attacker and defender for saving attributes to files
    return attacker, defender


def train_HEU_in_parallel(num_of_runs=100, episodes=50):
    '''
    Train the HEU in parallel. Run function 'train_for_HEU_distribution' with multiprocessing package.
    :param num_of_runs: number of runs in parallel
    :param episodes: number of episodes in each run
    :return:
    '''

    # Avoid the number of process is larger than the number of CPU cores
    # get the number of CPU cores
    num_of_cores = mp.cpu_count()
    # divide num_of_runs by the number of CPU cores, and save the result to a list
    num_of_runs_list = []
    for _ in range(num_of_runs // num_of_cores):
        num_of_runs_list.append(num_of_cores)
    if num_of_runs % num_of_cores:
        num_of_runs_list.append(num_of_runs % num_of_cores)

    # train the HEU in parallel
    all_results = []
    # a eisode counter for multiprocessing

    for num_of_process in num_of_runs_list:
        # create a pool of processes
        pool = mp.Pool(processes=num_of_process)
        # train the HEU in parallel
        results = [pool.apply_async(train_for_HEU_distribution, args=(episodes, ID)) for ID in range(num_of_process)]
        # close the pool
        pool.close()
        # wait for all the processes to finish
        pool.join()
        # get the results
        all_results += [p.get() for p in results]
        print("Got {} results".format(len(all_results)))
        print("=====================================")


    # get the attackers and defenders
    attackers = [res[0] for res in all_results]
    defenders = [res[1] for res in all_results]

    att_attribute_dict = None
    def_attribute_dict = None
    # save agent's attribute to dictionary and calculate the mean of the results
    # For attacker
    for attacker in attackers:
        temp_att_attri = attacker.__dict__
        temp_att_attri.pop('env')
        temp_att_attri.pop('writer')
        temp_att_attri['HEU_prob_record'] = np.mean(temp_att_attri['HEU_prob_record'], axis=0) # average over episodes

        if att_attribute_dict is None:
            att_attribute_dict = temp_att_attri
        else:
            for key in att_attribute_dict.keys():
                att_attribute_dict[key] += temp_att_attri[key]
    for key in att_attribute_dict.keys():
        att_attribute_dict[key] /= num_of_runs
    # force some attributes to be integer
    att_attribute_dict['self_strategy'] = int(att_attribute_dict['self_strategy'])
    att_attribute_dict['oppo_strategy'] = int(att_attribute_dict['oppo_strategy'])
    att_attribute_dict['subgame'] = int(att_attribute_dict['subgame'])
    # For defender
    for defender in defenders:
        temp_def_attri = defender.__dict__
        temp_def_attri.pop('env')
        temp_def_attri.pop('writer')
        temp_def_attri['HEU_prob_record'] = np.mean(temp_def_attri['HEU_prob_record'], axis=0)  # average over episodes

        if def_attribute_dict is None:
            def_attribute_dict = temp_def_attri
        else:
            for key in def_attribute_dict.keys():
                def_attribute_dict[key] += temp_def_attri[key]
    for key in def_attribute_dict.keys():
        def_attribute_dict[key] /= num_of_runs
    # force some attributes to be integer
    def_attribute_dict['self_strategy'] = int(def_attribute_dict['self_strategy'])
    def_attribute_dict['oppo_strategy'] = int(def_attribute_dict['oppo_strategy'])
    def_attribute_dict['subgame'] = int(def_attribute_dict['subgame'])

    print("attacker attribute: ", att_attribute_dict)
    print("defender attribute: ", def_attribute_dict)

    # get the HEU probability distribution result of attacker and defender, and saved to np.array
    att_HEU = np.array([attacker.HEU_prob_record for attacker in attackers])
    def_HEU = np.array([defender.HEU_prob_record for defender in defenders])
    # calculate the mean of the HEU probability distribution
    att_HEU_mean = np.mean(att_HEU, axis=0)
    def_HEU_mean = np.mean(def_HEU, axis=0)

    # get the action counter of attacker and defender, and saved to np.array
    att_action_counter = np.array([np.sum(attacker.action_counter, axis=1) for attacker in attackers])
    def_action_counter = np.array([np.sum(defender.action_counter, axis=1) for defender in defenders])
    # calculate the mean of the action counter
    att_action_counter_mean = np.mean(att_action_counter, axis=0)
    def_action_counter_mean = np.mean(def_action_counter, axis=0)

    # get the action counter in 2D, and saved to np.array
    att_action_counter_2D = np.array([attacker.action_counter for attacker in attackers])
    def_action_counter_2D = np.array([defender.action_counter for defender in defenders])
    # calculate the mean of the action counter in 2D
    att_action_counter_2D_mean = np.mean(att_action_counter_2D, axis=0)
    def_action_counter_2D_mean = np.mean(def_action_counter_2D, axis=0)

    # show the results
    print("=====================================")
    print("attacker HEU_prob: ", att_HEU)
    print("defender HEU_prob: ", def_HEU)
    print("=====================================")
    print("attacker HEU_prob mean: ", att_HEU_mean)
    print("defender HEU_prob mean: ", def_HEU_mean)
    print("=====================================")
    print("attacker action_counter: ", att_action_counter)
    print("defender action_counter: ", def_action_counter)
    print("=====================================")
    print("attacker att_action_counter_2D mean: ", att_action_counter_2D_mean)
    print("defender att_action_counter_2D mean: ", def_action_counter_2D_mean)

    # draw the results in one bar chart. x-axis is 0 to 9, y-axis is the probability of each action.
    # The attacker is blue and the defender is red. For each x-axis, attacker and defender are in the different bar.
    x = [i for i in range(len(att_HEU_mean))]
    # attacker
    plt.bar([i - 0.2 for i in x], att_HEU_mean, width=0.4, label="attacker", color="b")
    # defender
    plt.bar([i + 0.2 for i in x], def_HEU_mean, width=0.4, label="defender", color="r")
    plt.xlabel("Player's Strategy")
    plt.ylabel("HEU Probability Distribution (averaged over steps)")
    plt.xticks(x)
    plt.legend()
    plt.tight_layout()
    # save the figure
    file_path = "figures/"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    plt.savefig(file_path + "/HEU_distribution.png")
    plt.savefig(file_path + "/HEU_distribution.pdf")
    plt.show()

    # draw the results in one bar chart. x-axis is 0 to 9, y-axis is the counter of each action.
    # The attacker is blue and the defender is red. For each x-axis, attacker and defender are in the different bar.
    x = [i for i in range(len(att_action_counter_mean))]
    # attacker
    plt.bar([i - 0.2 for i in x], att_action_counter_mean, width=0.4, label="attacker", color="b")
    # defender
    plt.bar([i + 0.2 for i in x], def_action_counter_mean, width=0.4, label="defender", color="r")
    plt.xlabel("Player's Strategy")
    plt.ylabel("Action Counter (averaged over steps)")
    plt.xticks(x)
    plt.legend()
    plt.tight_layout()
    # save the figure
    plt.savefig(file_path + "/action_counter.png")
    plt.savefig(file_path + "/action_counter.pdf")
    plt.show()

    # draw the results in heatmap. x-axis is attack strategy from 0 to 9, y-axis is defense strategy from 0 to 9. The color is counter of each action.
    plt.imshow(att_action_counter_2D_mean, cmap="hot", interpolation="nearest")
    plt.ylabel("Attack Strategy (self)")
    plt.xlabel("Defense Strategy (opponent)")
    plt.xticks(x)
    plt.yticks(x)
    plt.colorbar()
    plt.tight_layout()
    # save the figure
    plt.savefig(file_path + "/action_counter_heatmap_att.png")
    plt.savefig(file_path + "/action_counter_heatmap_att.pdf")
    plt.show()

    plt.imshow(def_action_counter_2D_mean, cmap="hot", interpolation="nearest")
    plt.ylabel("Defense Strategy (self)")
    plt.xlabel("Attack Strategy (opponent)")
    plt.xticks(x)
    plt.yticks(x)
    plt.colorbar()
    plt.tight_layout()
    # save the figure
    plt.savefig(file_path + "/action_counter_heatmap_def.png")
    plt.savefig(file_path + "/action_counter_heatmap_def.pdf")
    plt.show()




    # save the mean of the results to files using pickle
    file_path = "data/HEU/"
    # create folders if not exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # save the results
    with open(file_path + "att_HEU_mean.pkl", "wb") as f:
        pickle.dump(att_HEU_mean, f)
    with open(file_path + "def_HEU_mean.pkl", "wb") as f:
        pickle.dump(def_HEU_mean, f)
    # save the agent's attribute to files using pickle
    with open(file_path + "att_attribute_dict.pkl", "wb") as f:
        pickle.dump(att_attribute_dict, f)
    with open(file_path + "def_attribute_dict.pkl", "wb") as f:
        pickle.dump(def_attribute_dict, f)



if __name__ == '__main__':
    train_HEU_in_parallel(num_of_runs=100, episodes=50)







