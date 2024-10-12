'''
Project     ：gym-drones
File        ：A3C_train_agent_optuna.py
Author      ：Zelin Wan
Date        ：8/15/22
Description : run this code to train defender or attacker model. Change variable 'is_defender' to choose defender or
attacker.
'''

import os
import random
import sys

import gym
import numpy as np

os.environ["OMP_NUM_THREADS"] = "1" # Error #34: System unable to allocate necessary resources for OMP thread:"

import time
import optuna
import torch

from sys import platform
from A3C_model import *
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR
# from Gym_HoneyDrone_Defender_and_Attacker import HyperGameSim


def objective(trial, fixed_seed=True, on_server=True):
    start_time = time.time()
    if is_defender:
        player_name = 'def'
    else:
        player_name = 'att'

    if trial is not None:
        trial_num_str = str(trial.number)
    else:
        trial_num_str = "None"
    # writer_hparam = SummaryWriter("runs/each_run_" +str(start_time) + "-" + player_name + "-" + "-Trial_" + trial_num_str + "-hparm")

    # This Configuration May Be Changed By Optuna:
    # glob_episode_thred: total number episode runs,
    # min_episode: minimum number of episode allowed to run before optuna pruning,
    # gamma: used to discount future reward, lr: learning rate, LR_decay: learning rate decay,
    # epsilon: probability of doing random action, epsilon_decay: epsilon decay,
    # pi_net_struc: structure of policy network, v_net_struct: structure of value network.
    config = dict(glob_episode_thred=10000, min_episode=1000, gamma=0.99, lr=0.001, LR_decay=0.95, epsilon = 0.1,
                  epsilon_decay = 0.9, pi_net_struc=[32, 128, 32], v_net_struct=[32, 128, 32])

    # 2. Suggest values of the hyperparameters using a trial object.
    if trial is not None:
        # config["glob_episode_thred"] = trial.suggest_int('glob_episode_thred', 1000, 1500, 100)     # total number of episodes
        config["gamma"] = trial.suggest_loguniform('gamma', 0.9, 0.99)
        config["lr"] = trial.suggest_loguniform('lr', 1e-4, 1e-1)
        config["LR_decay"] = trial.suggest_loguniform('LR_decay', 0.95, 0.999)   # since scheduler is not use. This one has no impact to reward
        config["epsilon"] = trial.suggest_loguniform('epsilon', 0.01, 0.5)
        # config["epsilon_decay"] = trial.suggest_loguniform('epsilon_decay', 0.9, 0.9)
        # network structure
        # pi_n_layers = trial.suggest_int('pi_n_layers', 3, 5)  # total number of layer
        # config["pi_net_struc"] = []     # Reset before append
        # for i in range(pi_n_layers):
        #     config["pi_net_struc"].append(trial.suggest_int(f'pi_n_units_l{i}', 32, 128, 32))   # try various nodes each layer
        # v_n_layers = trial.suggest_int('v_n_layers', 3, 5)  # total number of layer
        # config["v_net_struct"] = []     # Reset before append
        # for i in range(v_n_layers):
        #     config["v_net_struct"].append(trial.suggest_int(f'v_n_units_l{i}', 32, 128, 32))  # try various nodes each layer
    print("config", config)

    if on_server:
        num_worker = 128  # mp.cpu_count()     # update this for matching server's resources
    else:
        num_worker = 4

    # temp_env = HyperGameSim(fixed_seed=fixed_seed)
    # n_actions = temp_env.action_space[player_name].n
    # input_dims = temp_env.observation_space[player_name].shape
    # temp_env.close_env()    # close client for avoiding client limit error

    temp_env = gym.make('CartPole-v1')      # TODO: this is test
    n_actions = temp_env.action_space.n
    input_dims = temp_env.observation_space.shape

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Share Using", device)
    global_actor_critic = ActorCritic(input_dims, n_actions,
                                      lr=config['lr'],
                                      LR_decay=config['LR_decay'],
                                      gamma=config["gamma"],
                                      epsilon=config["epsilon"],
                                      epsilon_decay=config["epsilon_decay"],
                                      pi_net_struc=config["pi_net_struc"],
                                      v_net_struct=config["v_net_struct"],
                                      trial=trial,
                                      fixed_seed=fixed_seed).to(device)  # global NN
    print("global_actor_critic", global_actor_critic)
    global_actor_critic.share_memory()
    # optim = SharedAdam(global_actor_critic.parameters(), lr=config['lr'], betas=(0.9, 0.999))
    # global_ep = mp.Value('i', 0)

    # def lambda_function(epoch):  # epoch increase one when scheduler.step() is called
    #     return config["LR_decay"] ** epoch
    #
    # scheduler = LambdaLR(optim, lr_lambda=lambda_function)
    # scheduler = None    # don't use scheduler



    shared_dict = {}
    shared_dict["reward"] = Manager().list()  # use Manager().list() to create a shared list between processes
    shared_dict["t_step"] = Manager().list()
    shared_dict["score"] = Manager().list()
    shared_dict["oppo_score"] = Manager().list()
    shared_dict["lr"] = Manager().list()
    shared_dict["epsilon_in_choose_action"] = config["epsilon"]
    shared_dict["epsilon"] = Manager().list()
    shared_dict["eps"] = Manager().list()
    shared_dict["index"] = 0
    shared_dict["action"] = mp.Array('i', n_actions)
    shared_dict["start_time"] = str(start_time)
    shared_dict["ave_10_per_return"] = Manager().list()
    shared_dict["fixed_seed"] = fixed_seed
    shared_dict["on_server"] = on_server
    # shared_dict["reward"] = mp.Array('d', glob_episode_thred + num_worker)        # save simulation data ('d' means double-type)

    workers = [Agent(global_actor_critic,
                     input_dims,
                     n_actions,
                     name=i,
                     global_dict=shared_dict,
                     config=config,
                     glob_episode_thred=config['glob_episode_thred'],
                     player=player_name) for i in range(num_worker)]
    # workers = []
    # for i in range(num_worker):
    #     if i == 0:
    #         workers.append(Agent(global_actor_critic,
    #               input_dims,
    #               n_actions,
    #               name=i,
    #               global_dict=shared_dict,
    #               config=config,
    #               glob_episode_thred=config['glob_episode_thred'],
    #               player=player_name))
    #     else:
    #         workers.append(Agent(global_actor_critic,
    #               input_dims,
    #               n_actions,
    #               name=i,
    #               global_dict=shared_dict,
    #               config=config,
    #               glob_episode_thred=config['glob_episode_thred'],
    #               player=player_name))

    [w.start() for w in workers]
    [w.join() for w in workers]

    print("--- Simulation Time: %s seconds ---" % round(time.time() - start_time, 1))

    global_reward_10_per = [ele for ele in shared_dict["ave_10_per_return"]]    # get reward of all local agents
    if len(global_reward_10_per):
        ave_global_reward_10_per = sum(global_reward_10_per)/len(global_reward_10_per)
    else:
        ave_global_reward_10_per = 0

    # ========= Save global model =========
    # run 'tensorboard --logdir=runs' in terminal to start TensorBoard.
    if on_server:
        path = "/home/zelin/Drone/code_files/data/"+player_name
    else:
        path = "/Users/wanzelin/办公/gym-drones/data/A3C/"+player_name
    os.makedirs(path + "/model", exist_ok=True)
    torch.save(global_actor_critic.state_dict(), path + "/model/trained_A3C_" + str(start_time) + "_" + player_name + "_Trial_" + trial_num_str)

    # Write hparameter to tensorboard
    # convert list in self.config to integers
    temp_config = {}
    for key, value in config.items():
        if key == 'pi_net_struc':
            temp_config['pi_net_num'] = len(value)
            for index, num_node in enumerate(value):
                temp_config['pi_net' + str(index)] = num_node
        elif key == 'v_net_struct':
            temp_config['v_net_num'] = len(value)
            for index, num_node in enumerate(value):
                temp_config['v_net' + str(index)] = num_node
        else:
            temp_config[key] = value
    if on_server:
        path = "/home/zelin/Drone/code_files/data/"
    else:
        path = ""
    writer_hparam = SummaryWriter(path + "runs_" + player_name + "/each_run_" + str(start_time) + "-" + player_name +
                                  "-Trial_" + trial_num_str + "-hparm")

    writer_hparam.add_hparams(temp_config, {'return_reward': ave_global_reward_10_per})  # add for Hyperparameter Tuning
    writer_hparam.flush()
    writer_hparam.close()

    return ave_global_reward_10_per  # return average value


if __name__ == '__main__':
    is_defender = True  # True means train a defender RL, False means train an attacker RL
    test_mode = True  # True means use preset hyperparameter, and optuna will not be used.
    fixed_seed = False # True means the seeds for pytorch, numpy, and python will be fixed.

    if is_defender:
        player_name = 'def'
        print("running for defender")
    else:
        player_name = 'att'
        print("running for attacker")


    if fixed_seed:
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

    if platform == "darwin":
        on_server = False
    else:
        on_server = True
    # objective(None)
    # 3. Create a study object and optimize the objective function.
    # /home/zelin/Drone/data
    if test_mode:
        objective(None, fixed_seed=fixed_seed, on_server=on_server)
        print("testing mode")
    else:
        print("training mode")
        if on_server:
            db_path = "/home/zelin/Drone/code_files/data/"+player_name+"/"
            os.makedirs(db_path, exist_ok=True)
            study = optuna.create_study(direction='maximize', study_name="A3C-hyperparameter-study",
                                        storage="sqlite://///"+db_path+"HyperPara_database.db",
                                        load_if_exists=True)
        else:
            db_path = "/Users/wanzelin/办公/gym-drones/data/"+player_name+"/"
            os.makedirs(db_path, exist_ok=True)
            study = optuna.create_study(direction='maximize', study_name="A3C-hyperparameter-study",
                                        storage="sqlite:////"+db_path+"HyperPara_database.db",
                                        load_if_exists=True)
        study.optimize(lambda trial: objective(trial, fixed_seed, on_server), n_trials=100)
