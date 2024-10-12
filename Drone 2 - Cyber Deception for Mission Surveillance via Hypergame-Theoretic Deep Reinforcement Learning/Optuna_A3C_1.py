'''
Project     ：Drone-DRL-HT 
File        ：Optuna_A3C_1.py
Author      ：Zelin Wan
Date        ：2/8/23
Description : 
'''
import sys
import argparse
import gym
import os

os.environ["OMP_NUM_THREADS"] = "1"  # Error #34: System unable to allocate necessary resources for OMP thread:"
import torch.multiprocessing as mp
import numpy as np
import optuna
import random
import time
import pickle

from shared_classes import SharedAdam, SharedSGD
from A3C_model_1 import *
from sys import platform
from multiprocessing import Manager
from utils import get_last_ten_ave
from torch.utils.tensorboard import SummaryWriter
from Gym_Defender_and_Attacker import HyperGameSim
from datetime import datetime


def objective(trial, fixed_seed=True, on_server=True, is_custom_env=True, def_select_method='fixed', att_select_method='fixed', miss_dur=200, target_size=5, max_att_budget=5,
              num_HD=5,
              test_mode=False, epsilon=0.0):
    start_time = time.time()

    if trial is not None:
        trial_num_str = str(trial.number)
    else:
        trial_num_str = "None"

    # This Configuration May Be Changed By Optuna:
    # glob_episode_thred: total number episode runs,
    # min_episode: minimum number of episode allowed to run before optuna pruning,
    # gamma: used to discount future reward, lr: learning rate, LR_decay: learning rate decay,
    # epsilon: probability of doing random action, epsilon_decay: epsilon decay,
    # pi_net_struc: structure of policy network, v_net_struct: structure of value network.

    # default setting for players
    glob_episode = 260 #100 #600 #1000
    # default setting for defender
    config_def = dict(glob_episode_thred=glob_episode, min_episode=1000, gamma=0.99,
                      lr=0.0005, LR_decay=0.9, epsilon=epsilon,
                      epsilon_decay=0.99, pi_net_struc=[128, 128, 64, 32], v_net_struct=[128, 128, 64, 32])

    # default setting for attacker
    config_att = dict(glob_episode_thred=glob_episode, min_episode=1000, gamma=0.99,
                      lr=0.0005, LR_decay=0.9, epsilon=epsilon,
                      epsilon_decay=0.99, pi_net_struc=[128, 128, 64, 32], v_net_struct=[128, 128, 64, 32])

    # Suggest values of the hyperparameters using a trial object.
    if trial is not None:
        if def_select_method in ['DRL', 'HT-DRL']:
        # if exp_scheme == 0 or exp_scheme == 2:  # those two scheme use D-a3c
            # trial for defender
            config_def["gamma"] = trial.suggest_loguniform('gamma_def', 0.5, 0.999)
            config_def["lr"] = trial.suggest_loguniform('lr_def', 0.0001, 0.01)
            config_def["LR_decay"] = trial.suggest_loguniform('LR_decay_def', 0.9, 0.999)
            # config_def["epsilon"] = trial.suggest_loguniform('epsilon_def', 0.01, 1.0)
            config_def["epsilon_decay"] = trial.suggest_loguniform('epsilon_decay_def', 0.9, 0.999)

        if att_select_method in ['DRL', 'HT-DRL']:
        # if exp_scheme == 1 or exp_scheme == 2:  # those two scheme use A-a3c
            # trial for attacker
            config_att["gamma"] = trial.suggest_loguniform('gamma_att', 0.5, 0.999)
            config_att["lr"] = trial.suggest_loguniform('lr_att', 0.0001, 0.01)
            config_att["LR_decay"] = trial.suggest_loguniform('LR_decay_att', 0.9, 0.999)
            # config_att["epsilon"] = trial.suggest_loguniform('epsilon_att', 0.01, 1.0)
            config_att["epsilon_decay"] = trial.suggest_loguniform('epsilon_decay_att', 0.9, 0.999)

    print("config_def", config_def)
    print("config_att", config_att)

    # When defender uses No-Defense, IDS, or CD, disable all Honey Drones.
    if def_select_method in ['No-Defense', 'IDS', 'CD']:
        num_HD = 0

    if is_custom_env:
        env = HyperGameSim(fixed_seed=fixed_seed, miss_dur=miss_dur, target_size=target_size,
                           max_att_budget=max_att_budget, num_HD=num_HD, def_select_method=def_select_method)
        input_dims_def = env.observation_space['def'].shape[0]
        n_actions_def = env.action_space['def'].n
        input_dims_att = env.observation_space['att'].shape[0]
        n_actions_att = env.action_space['att'].n
    else:
        show_env = False
        if show_env:
            env = gym.make('CartPole-v1', render_mode='human')
        else:
            env = gym.make('CartPole-v1')
        input_dims_def = env.observation_space.shape[0]
        n_actions_def = env.action_space.n
        input_dims_att = 1  # att network will not be used
        n_actions_att = 1  # att network will not be used

    # read HEU distribution
    file_path = "data/HEU/"
    # HEU_guilded_def/att: the ratio of HEU to be used in the game
    HEU_guilded_def = 0.0
    HEU_guilded_att = 0.0
    # if not using HEU to guild DRL, set the initial probability distribution to None
    att_HEU_mean = None
    def_HEU_mean = None
    if def_select_method in ['HEU', 'HT-DRL']:
        HEU_guilded_def = 1.0
        # read the 'def_HEU_mean.pkl' for initial probability distribution
        def_HEU_mean = pickle.load(open(file_path + "def_HEU_mean.pkl", "rb"))
        print("def_HEU_mean", def_HEU_mean)

    if att_select_method in ['HEU', 'HT-DRL']:
        HEU_guilded_att = 1.0
        # read the 'att_HEU_mean.pkl' for initial probability distribution
        att_HEU_mean = pickle.load(open(file_path + "att_HEU_mean.pkl", "rb"))
        print("att_HEU_mean", att_HEU_mean)

    # read 'att_attribute_dict.pkl' and 'def_attribute_dict.pkl' to create HEU agent that provides reward
    att_attribute_dict = pickle.load(open(file_path + "att_attribute_dict.pkl", "rb"))
    def_attribute_dict = pickle.load(open(file_path + "def_attribute_dict.pkl", "rb"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Share Using", device)

    # defender's global AC
    glob_AC_def = ActorCritic(input_dims=input_dims_def, n_actions=n_actions_def, epsilon=config_def["epsilon"],
                              epsilon_decay=config_def["epsilon_decay"], pi_net_struc=config_def["pi_net_struc"],
                              v_net_struct=config_def["v_net_struct"],
                              HEU_prob_distribution=def_HEU_mean, HEU_guilded=HEU_guilded_def).to(device)  # Global Actor Critic

    print("global_actor_critic", glob_AC_def)
    glob_AC_def.share_memory()  # share the global parameters in multiprocessing
    optim_def = SharedAdam(glob_AC_def.parameters(), lr=config_def["lr"])  # global optimizer

    # attacker's global AC
    glob_AC_att = ActorCritic(input_dims=input_dims_att, n_actions=n_actions_att, epsilon=config_att["epsilon"],
                              epsilon_decay=config_att["epsilon_decay"], pi_net_struc=config_att["pi_net_struc"],
                              v_net_struct=config_att["v_net_struct"],
                              HEU_prob_distribution=att_HEU_mean, HEU_guilded=HEU_guilded_att).to(device)  # Global Actor Critic

    print("global_actor_critic", glob_AC_att)
    glob_AC_att.share_memory()  # share the global parameters in multiprocessing
    optim_att = SharedAdam(glob_AC_att.parameters(), lr=config_att["lr"])  # global optimizer

    # shared_dict is shared by all parallel processes
    shared_dict = {}
    shared_dict['global_ep'] = mp.Value('i', 0)
    shared_dict['glob_r_list'] = Manager().list()
    shared_dict["start_time"] = str(start_time)
    shared_dict['eps_writer'] = mp.Queue()  # '_writer' means it will be used for tensorboard writer
    shared_dict['score_def_writer'] = mp.Queue()
    shared_dict['score_att_writer'] = mp.Queue()
    shared_dict['score_def_avg_writer'] = mp.Queue()
    shared_dict['score_att_avg_writer'] = mp.Queue()
    shared_dict['def_stra_count_writer'] = mp.Queue()
    shared_dict['att_stra_count_writer'] = mp.Queue()
    shared_dict['lr_writer'] = mp.Queue()
    shared_dict['lr_def_writer'] = mp.Queue()
    shared_dict['lr_att_writer'] = mp.Queue()
    shared_dict["epsilon_writer"] = mp.Queue()
    shared_dict["epsilon_def_writer"] = mp.Queue()
    shared_dict["epsilon_att_writer"] = mp.Queue()
    shared_dict['t_loss_writer'] = mp.Queue()
    shared_dict['t_loss_writer'] = mp.Queue()
    shared_dict['c_loss_writer'] = mp.Queue()
    shared_dict['a_loss_writer'] = mp.Queue()
    shared_dict['t_loss_def_writer'] = mp.Queue()  # total loss of actor critic for defender
    shared_dict['c_loss_def_writer'] = mp.Queue()  # loss of critic for defender
    shared_dict['a_loss_def_writer'] = mp.Queue()  # loss of actor for defender
    shared_dict['t_loss_att_writer'] = mp.Queue()  # total loss of actor critic for attacker
    shared_dict['c_loss_att_writer'] = mp.Queue()  # loss of critic for attacker
    shared_dict['a_loss_att_writer'] = mp.Queue()  # loss of actor for attacker
    shared_dict['mission_condition_writer'] = mp.Queue()  # 0 means mission fail, 1 means mission success
    shared_dict['total_energy_consump_writer'] = mp.Queue()  # energy consumption of drones
    shared_dict['scan_percent_writer'] = mp.Queue()  # percentage of scanned cell
    shared_dict['MDHD_active_num_writer'] = mp.Queue()  #
    shared_dict['MDHD_connect_RLD_num_writer'] = mp.Queue()  #
    shared_dict['MDHD_connect_RLD_num_writer'] = mp.Queue()
    shared_dict['remaining_time_writer'] = mp.Queue()
    shared_dict['energy_HD_writer'] = mp.Queue()
    shared_dict['energy_MD_writer'] = mp.Queue()
    shared_dict['on_server'] = on_server
    shared_dict['def_HEU_decision_ratio'] = mp.Queue()
    shared_dict['att_HEU_decision_ratio'] = mp.Queue()
    shared_dict['def_random_decision_ratio'] = mp.Queue()
    shared_dict['att_random_decision_ratio'] = mp.Queue()
    shared_dict['def_DRL_decision_ratio'] = mp.Queue()
    shared_dict['att_DRL_decision_ratio'] = mp.Queue()
    # global_ep_r = mp.Value('d', 0.)

    # parallel training
    if on_server:
        num_worker = 1  # mp.cpu_count()       # update this for matching server's resources (ARC can hold up to 75
        # workers if allocate 128 CPUs)
    else:
        num_worker = 1

    # run Agent in parallel
    workers = [Agent(glob_AC_def, glob_AC_att, optim_def, optim_att, shared_dict=shared_dict, config_def=config_def,
                     config_att=config_att, def_attri_dict=def_attribute_dict, att_attri_dict=att_attribute_dict,
                     fixed_seed=fixed_seed, trial=trial,
                     name_id=i, def_select_method=def_select_method, att_select_method=att_select_method,
                     is_custom_env=is_custom_env, miss_dur=miss_dur,
                     target_size=target_size, max_att_budget=max_att_budget, num_HD=num_HD, epsilon=epsilon) for i in range(num_worker)]

    print("start workers")
    [w.start() for w in workers]

    print("join workers")
    [w.join() for w in workers]

    print("--- Simulation Time: %s seconds ---" % round(time.time() - start_time, 1))

    # ========= Save Data for Optuna =========
    score_list = [ele for ele in shared_dict['glob_r_list']]  # get reward of all local agents
    last_10_per_reward_mean = get_last_ten_ave(score_list)

    # ========= Save global model =========
    # run 'tensorboard --logdir=runs' in terminal to start TensorBoard.
    if on_server:
        path = "/projects/zelin1/Drone_DRL_HT/data/" + str(miss_dur) + "_" + str(max_att_budget) + "_" + str(
            num_HD) + "/model/" + scheme_name
    else:
        path = "/Users/wanzelin/办公/Drone_DRL_HT/data/" + str(miss_dur) + "_" + str(max_att_budget) + "_" + str(
            num_HD) + "/model/" + scheme_name
    os.makedirs(path, exist_ok=True)
    torch.save(glob_AC_def.state_dict(),
               path + "/trained_A3C_" + str(start_time) + "_" + scheme_name + "_Trial_" + trial_num_str)

    # ========= Write Hparameter to Tensorboard =========
    # convert list in self.config to integers
    temp_config = {}
    if def_select_method in ['HEU', 'DLR', 'HT-DRL']:
        items_for_TB_hpara = config_def.items()
    else:
        items_for_TB_hpara = config_att.items()
    # if exp_scheme == 1:  # case of find attacker's hyperparameter
    #     items_for_TB_hpara = config_att.items()
    # else:  # all other case only consider defender's hyperparameter
    #     items_for_TB_hpara = config_def.items()
    for key, value in items_for_TB_hpara:
        if key == 'pi_net_struc':
            # loop write structure
            temp_config['pi_net_num'] = len(value)
            for index, num_node in enumerate(value):
                temp_config['pi_net' + str(index)] = num_node
        elif key == 'v_net_struct':
            # loop write structure
            temp_config['v_net_num'] = len(value)
            for index, num_node in enumerate(value):
                temp_config['v_net' + str(index)] = num_node
        else:
            # simply write down
            temp_config[key] = value

    # write hyperparameter to tensorboard
    if test_mode==False:
        if on_server:
            path = "/projects/zelin1/Drone_DRL_HT/data/" + str(miss_dur) + "_" + str(max_att_budget) + "_" + str(num_HD) + "/"
        else:
            path = "data/" + str(miss_dur) + "_" + str(max_att_budget) + "_" + str(num_HD) + "/"
        print("create writer")
        current_time = datetime.now().strftime("%Y-%m-%d-%H_%M_%S-%f")
        writer_hparam = SummaryWriter(
            log_dir=path + "scheme_" + scheme_name + "/each_run-" + "scheme_" + scheme_name + "-Time_" + current_time
                    + "-Trial_" + trial_num_str + "-hparm")
        print("add hparams")
        print("last_10_per_reward_mean: ", last_10_per_reward_mean)
        writer_hparam.add_hparams(temp_config, {'return_reward': last_10_per_reward_mean})  # add for Hyperparameter Tuning
        print("writer_hparam.flush")
        writer_hparam.flush()
        print("writer_hparam.close")
        writer_hparam.close()

    return last_10_per_reward_mean  # return average value

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    # ========= Set up parameters =========
    # exp_scheme = 2  # 0 means A-random D-a3c, 1 means A-a3c D-random, 2 means A-a3c D-a3c, 3 means A-random D-random
    test_mode = True  # True means use preset hyperparameter, and optuna will not be used. False means use optuna
    att_select_method = 'HEU' # ['fixed', 'HEU', 'DRL', 'HT-DRL']
    def_select_method = 'HEU'  # ['fixed', 'HEU', 'DRL', 'HT-DRL', 'No-Defense', 'IDS', 'CD']
    fixed_seed = False  # True means the seeds for pytorch, numpy, and python will be fixed.
    is_custom_env = True  # True means use the customized drone environment, False means use gym 'CartPole-v1'.

    # sensitivity analysis value
    miss_dur = 150 # default: 30. Try to change from 10, 20, 30, 40 to 50
    max_att_budget = 5  # default: 5. The maximum number of attack can launch in a round
    num_HD = 5  # default: 2. The number of honey drone
    target_size = 5  # default: 5. The 'Number of Cell to Scan' = 'target_size' * 'target_size'

    test_mode_run_time = 1  # 50  # default: 50. The number of times to run the test mode
    # =====================================

    # ======== Read parameters from command line ========
    parser = argparse.ArgumentParser()
    # parser.add_argument("--exp_scheme", type=int, default=exp_scheme)
    parser.add_argument("--att_select_method", type=str, default=att_select_method)
    parser.add_argument("--def_select_method", type=str, default=def_select_method)
    parser.add_argument("--fixed_seed", type=str2bool, default=fixed_seed)
    parser.add_argument("--miss_dur", type=int, default=miss_dur)
    parser.add_argument("--max_att_budget", type=int, default=max_att_budget)
    parser.add_argument("--num_HD", type=int, default=num_HD)
    parser.add_argument("--target_size", type=int, default=target_size)
    parser.add_argument("--run_time", type=int, default=test_mode_run_time)

    args = parser.parse_args()
    # exp_scheme = args.exp_scheme
    att_select_method = args.att_select_method
    def_select_method = args.def_select_method
    fixed_seed = args.fixed_seed
    print("args", args)
    miss_dur = args.miss_dur
    max_att_budget = args.max_att_budget
    num_HD = args.num_HD
    target_size = args.target_size
    test_mode_run_time = args.run_time
    # ===================================================
    scheme_name = att_select_method + '-' + def_select_method

    # check if the input is valid
    if def_select_method not in ['fixed', 'HEU', 'DRL', 'HT-DRL', 'No-Defense', 'IDS', 'CD']:
        raise Exception("def_select_method is not valid")
    if att_select_method not in ['fixed', 'HEU', 'DRL', 'HT-DRL']:
        raise Exception("att_select_method is not valid")


    # if exp_scheme == 0:
    #     player_name = 'def'
    #     print("running for defender")
    # elif exp_scheme == 1:
    #     player_name = 'att'
    #     print("running for attacker")
    # elif exp_scheme == 2:
    #     player_name = 'DefAtt'
    #     print("running for defender and attacker")
    # elif exp_scheme == 3:
    #     player_name = 'random'
    #     fixed_seed = False
    #     print("running randomly for defender and attacker")
    # else:
    #     raise Exception("invalid exp_scheme, see comment")

    if fixed_seed:
        # torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

    if platform == "darwin":
        on_server = False
    else:
        on_server = True
    # objective(None)
    # 3. Create a study object and optimize the objective function.
    # /projects/zelin1/Drone/data
    if test_mode:
        print("testing mode")
        for _ in range(test_mode_run_time):
            objective(None, fixed_seed=fixed_seed, on_server=on_server, is_custom_env=is_custom_env,
                      def_select_method=def_select_method, att_select_method=att_select_method,
                      miss_dur=miss_dur, target_size=target_size,
                      max_att_budget=max_att_budget, num_HD=num_HD, test_mode=test_mode)
    else:
        print("training mode")
        if on_server:
            db_path = "/projects/zelin1/Drone_DRL_HT/data/" + str(miss_dur) + "_" + str(max_att_budget) + "_" + str(
                num_HD) + scheme_name + "/"
            os.makedirs(db_path, exist_ok=True)
            study = optuna.create_study(direction='maximize', study_name="A3C-hyperparameter-study",
                                        storage="sqlite://///" + db_path + "HyperPara_database_sche-" + scheme_name + ".db",
                                        load_if_exists=True)
        else:
            code_file_path = os.getcwd()
            db_path = code_file_path + "/data/" + str(miss_dur) + "_" + str(max_att_budget) + "_" + str(
                num_HD) + "/" + scheme_name + "/"
            os.makedirs(db_path, exist_ok=True)
            study = optuna.create_study(direction='maximize', study_name="A3C-hyperparameter-study",
                                        storage="sqlite:////" + db_path + "HyperPara_database_sche-" + scheme_name + ".db",
                                        load_if_exists=True)
        study.optimize(
            lambda trial: objective(trial, fixed_seed, on_server, is_custom_env, def_select_method, att_select_method,
                                    miss_dur, target_size, max_att_budget, num_HD), n_trials=50)
