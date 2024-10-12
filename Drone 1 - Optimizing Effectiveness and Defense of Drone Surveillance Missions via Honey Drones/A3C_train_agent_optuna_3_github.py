'''
Project     ：gym-drones 
File        ：A3C_train_agent_optuna_3_github.py
Author      ：Zelin Wan
Date        ：10/10/22
Description : 
'''

import os
os.environ["OMP_NUM_THREADS"] = "1"
import torch.multiprocessing as mp
import numpy as np
import optuna
import random
import time

from shared_adam import SharedAdam
from A3C_model_3_github import *
from sys import platform
from multiprocessing import Manager
from utils import get_last_ten_ave
from torch.utils.tensorboard import SummaryWriter
from Gym_HoneyDrone_Defender_and_Attacker import HyperGameSim



def objective(trial, fixed_seed=True, on_server=True, is_defender=True, is_custom_env=True, miss_dur=30, target_size=5):
    start_time = time.time()
    if is_defender:
        player_name = 'def'
    else:
        player_name = 'att'

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
    if is_defender:
        # default setting for defender
        config = dict(glob_episode_thred=5000, min_episode=1000, gamma=0.913763771439141, lr=0.00135463670164839,
                      LR_decay=0.97127344458321, epsilon=0.0119790647086112,
                      epsilon_decay=0.970168515121627, pi_net_struc=[64, 128, 128, 64], v_net_struct=[64, 128, 128, 64])
    else:
        # default setting for attacker
        config = dict(glob_episode_thred=5000, min_episode=1000, gamma=0.988784695574701, lr=0.00861718519746169,
                      LR_decay=0.974819836482024, epsilon=0.0642399295782559,
                      epsilon_decay=0.961601926251759, pi_net_struc=[64, 128, 128, 64], v_net_struct=[64, 128, 128, 64])

    # Suggest values of the hyperparameters using a trial object.
    if trial is not None:
        config["gamma"] = trial.suggest_loguniform('gamma', 0.9, 0.99)
        config["lr"] = trial.suggest_loguniform('lr', 0.001, 0.01)
        config["LR_decay"] = trial.suggest_loguniform('LR_decay', 0.9, 0.99)  # since scheduler is not use. This one has no impact to reward
        config["epsilon"] = trial.suggest_loguniform('epsilon', 0.01, 0.5)
        config["epsilon_decay"] = trial.suggest_loguniform('epsilon_decay', 0.9, 0.99)
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

    if is_custom_env:
        env = HyperGameSim(fixed_seed=fixed_seed)
        input_dims = env.observation_space[player_name].shape[0]
        n_actions = env.action_space[player_name].n
        env.close_env()
    else:
        env = gym.make('CartPole-v1')
        input_dims = env.observation_space.shape[0]
        # input_dims = env.observation_space.n
        n_actions = env.action_space.n


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Share Using", device)

    print("config epsilon", config["epsilon"])
    glob_AC = ActorCritic(input_dims=input_dims, n_actions=n_actions, epsilon=config["epsilon"],
                          epsilon_decay=config["epsilon_decay"], pi_net_struc=config["pi_net_struc"],
                          v_net_struct=config["v_net_struct"], fixed_seed=fixed_seed).to(device)  # Global Actor Critic
    print("global_actor_critic", glob_AC)
    glob_AC.share_memory()  # share the global parameters in multiprocessing
    optim = SharedAdam(glob_AC.parameters(), lr=config["lr"])  # global optimizer

    # shared_dict is shared by all parallel processes
    shared_dict = {}
    shared_dict['global_ep'] = mp.Value('i', 0)
    shared_dict['glob_r_list'] = Manager().list()
    shared_dict["start_time"] = str(start_time)
    shared_dict['eps_writer'] = mp.Queue()      # '_writer' means it will be used for tensorboard writer
    shared_dict['score_def_writer'] = mp.Queue()
    shared_dict['score_att_writer'] = mp.Queue()
    shared_dict['score_def_avg_writer'] = mp.Queue()
    shared_dict['score_att_avg_writer'] = mp.Queue()
    shared_dict['def_stra_count_writer'] = mp.Queue()
    shared_dict['att_stra_count_writer'] = mp.Queue()
    shared_dict['lr_writer'] = mp.Queue()
    shared_dict["epsilon_writer"] = mp.Queue()
    shared_dict['t_loss_writer'] = mp.Queue()   # total loss of actor critic
    shared_dict['c_loss_writer'] = mp.Queue()   # loss of critic
    shared_dict['a_loss_writer'] = mp.Queue()   # loss of actor
    shared_dict['att_succ_rate_writer'] = mp.Queue()    # attack success rate
    shared_dict['mission_condition_writer'] = mp.Queue()   # 0 means mission fail, 1 means mission success
    shared_dict['total_energy_consump_writer'] = mp.Queue()    # energy consumption of drones
    shared_dict['scan_percent_writer'] = mp.Queue()     # percentage of scanned cell
    shared_dict['att_reward_0_writer'] = mp.Queue()       # the first component of attacker's reward
    shared_dict['att_reward_1_writer'] = mp.Queue()  # the second component of attacker's reward
    shared_dict['att_reward_2_writer'] = mp.Queue()  # the third component of attacker's reward
    shared_dict['def_reward_0_writer'] = mp.Queue()  # the first component of defender's reward
    shared_dict['def_reward_1_writer'] = mp.Queue()  # the second component of defender's reward
    shared_dict['def_reward_2_writer'] = mp.Queue()  # the third component of defender's reward
    shared_dict['MDHD_active_num_writer'] = mp.Queue()  #
    shared_dict['MDHD_connect_RLD_num_writer'] = mp.Queue()  #
    shared_dict['MDHD_connect_RLD_num_writer'] = mp.Queue()
    shared_dict['mission_complete_rate_writer'] = mp.Queue()
    shared_dict['remaining_time_writer'] = mp.Queue()
    shared_dict['energy_HD_writer'] = mp.Queue()
    shared_dict['energy_MD_writer'] = mp.Queue()
    shared_dict['on_server'] = on_server


    # global_ep_r = mp.Value('d', 0.)

    # parallel training
    if on_server:
        num_worker = 128  # mp.cpu_count()     # update this for matching server's resources
    else:
        num_worker = 9
    workers = [Agent(glob_AC, optim, shared_dict=shared_dict, lr_decay=config["LR_decay"], gamma=config["gamma"],
                     epsilon=config["epsilon"], epsilon_decay=config["epsilon_decay"],
                     MAX_EP=config["glob_episode_thred"], fixed_seed=fixed_seed, trial=trial,
                     name_id=i, player=player_name, is_custom_env=is_custom_env, miss_dur=miss_dur, target_size=target_size) for i in range(num_worker)]

    [w.start() for w in workers]
    [w.join() for w in workers]

    print("--- Simulation Time: %s seconds ---" % round(time.time() - start_time, 1))

    # ========= Save Data for Optuna =========
    score_list = [ele for ele in shared_dict['glob_r_list']]    # get reward of all local agents
    last_10_per_reward_mean = get_last_ten_ave(score_list)


    # ========= Save global model =========
    # run 'tensorboard --logdir=runs' in terminal to start TensorBoard.
    if on_server:
        path = "/home/zelin/Drone/data/"+str(miss_dur)+"_"+str(target_size)+"/"+ player_name+"/"
    else:
        path = "/Users/wanzelin/办公/gym-drones/data/"+str(miss_dur)+"_"+str(target_size)+"/"+player_name+"/"
    os.makedirs(path + "model", exist_ok=True)
    torch.save(glob_AC.state_dict(), path + "model/trained_A3C_" + str(start_time) + "_" + player_name + "_Trial_" + trial_num_str)


    # ========= Write Hparameter to Tensorboard =========
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
        path = "/home/zelin/Drone/data/" +str(miss_dur)+"_"+str(target_size)+"/"
    else:
        path = "data/" +str(miss_dur)+"_"+str(target_size)+"/"
    writer_hparam = SummaryWriter(log_dir=path + "runs_" + player_name + "/each_run_" + str(start_time) + "-" + player_name +
                                  "-Trial_" + trial_num_str + "-hparm")

    writer_hparam.add_hparams(temp_config, {'return_reward': last_10_per_reward_mean})  # add for Hyperparameter Tuning
    writer_hparam.flush()
    writer_hparam.close()

    return last_10_per_reward_mean  # return average value


if __name__ == '__main__':
    test_mode = True       # True means use preset hyperparameter, and optuna will not be used.
    is_defender = True      # True means train a defender RL, False means train an attacker RL
    fixed_seed = True       # True means the seeds for pytorch, numpy, and python will be fixed.
    is_custom_env = True    # True means use the customized drone environment, False means use gym 'CartPole-v1'.

    # sensitivity analysis value
    miss_dur = 30 # default: 30
    target_size = 5 # default: 5. The 'Number of Cell to Scan' = 'target_size' * 'target_size'

    test_mode_run_time = 10

    if is_defender:
        player_name = 'def'
        print("running for defender")
    else:
        player_name = 'att'
        print("running for attacker")


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
    # /home/zelin/Drone/data
    if test_mode:
        print("testing mode")
        for _ in range(test_mode_run_time):
            objective(None, fixed_seed=fixed_seed, on_server=on_server, is_defender=is_defender, is_custom_env=is_custom_env, miss_dur=miss_dur, target_size=target_size)
    else:
        print("training mode")
        if on_server:
            db_path = "/home/zelin/Drone/data/"+str(miss_dur)+"_"+str(target_size)+"/"+player_name+"/"
            os.makedirs(db_path, exist_ok=True)
            study = optuna.create_study(direction='maximize', study_name="A3C-hyperparameter-study",
                                        storage="sqlite://///"+db_path+"HyperPara_database.db",
                                        load_if_exists=True)
        else:
            db_path = "/Users/wanzelin/办公/gym-drones/data/"+str(miss_dur)+"_"+str(target_size)+"/"+player_name+"/"
            os.makedirs(db_path, exist_ok=True)
            study = optuna.create_study(direction='maximize', study_name="A3C-hyperparameter-study",
                                        storage="sqlite:////"+db_path+"HyperPara_database.db",
                                        load_if_exists=True)
        study.optimize(lambda trial: objective(trial, fixed_seed, on_server, is_defender, is_custom_env, miss_dur, target_size), n_trials=100)


