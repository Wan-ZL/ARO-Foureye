# Code is heavily inspired by Morvan Zhou's code. Please check out
# his work at github.com/MorvanZhou/pytorch-A3C


# TODO: Create attacker model (save)
# TODO: add two models to environment


import os
os.environ["OMP_NUM_THREADS"] = "1" # Error #34: System unable to allocate necessary resources for OMP thread:"

import ctypes

import pickle
import time
from copy import copy

import gym
import torch as torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from Gym_HoneyDrone_Defender_and_Attacker import HyperGameSim
from multiprocessing import Manager
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR
import optuna
from sys import platform
from A3C_model import *


# class SharedAdam(torch.optim.Adam):
#     def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
#                  weight_decay=0):
#         super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps,
#                                          weight_decay=weight_decay)
#
#         for group in self.param_groups:
#             for p in group['params']:
#                 state = self.state[p]
#                 state['step'] = 0
#                 state['exp_avg'] = torch.zeros_like(p.data)
#                 state['exp_avg_sq'] = torch.zeros_like(p.data)
#
#                 state['exp_avg'].share_memory_()
#                 state['exp_avg_sq'].share_memory_()


# class ActorCritic(nn.Module):
#     def __init__(self, input_dims, n_actions, gamma=0.99, pi_net_struc=[128], v_net_struct=[128]):
#         super(ActorCritic, self).__init__()
#
#         self.gamma = gamma
#         self.pi_net = self.build_Net(*input_dims, n_actions, pi_net_struc)
#         self.v_net = self.build_Net(*input_dims, 1, v_net_struct)
#
#         # self.pi1 = nn.Linear(*input_dims, 128)
#         # self.v1 = nn.Linear(*input_dims, 128)
#         # self.pi2 = nn.Linear(128, 256)
#         # self.v2 = nn.Linear(128, 256)
#         # self.pi3 = nn.Linear(256, 128)
#         # self.v3 = nn.Linear(256, 128)
#         # self.pi = nn.Linear(128, n_actions)
#         # self.v = nn.Linear(128, 1)
#
#         self.rewards = []
#         self.actions = []
#         self.states = []
#
#
#     def build_Net(self, obser_space, action_space, net_struc):
#         layers = []
#         in_features = obser_space
#         # for i in range(n_layers):
#         for node_num in net_struc:
#             layers.append(nn.Linear(in_features, node_num))
#             layers.append(nn.ReLU())
#             in_features = node_num
#         layers.append(nn.Linear(in_features, action_space))
#         net = nn.Sequential(*layers)
#         return net
#
#     def remember(self, state, action, reward):
#         self.states.append(state)
#         self.actions.append(action)
#         self.rewards.append(reward)
#
#     def clear_memory(self):
#         self.states = []
#         self.actions = []
#         self.rewards = []
#
#     # def forward(self, state):
#     #     pi1 = F.relu(self.pi1(state))
#     #     v1 = F.relu(self.v1(state))
#     #
#     #     pi2 = self.pi2(pi1)
#     #     v2 = self.v2(v1)
#     #
#     #     pi3 = F.relu(self.pi3(pi2))
#     #     v3 = F.relu(self.v3(v2))
#     #
#     #     pi = self.pi(pi3)
#     #     v = self.v(v3)
#     #
#     #     return pi, v
#
#     def calc_R(self, done):
#         states = torch.tensor(self.states, dtype=torch.float)
#         # _, v = self.forward(states)
#         v = self.v_net(states)
#
#         R = v[-1] * (1 - int(done))
#
#         batch_return = []
#         for reward in self.rewards[::-1]:
#             R = reward + self.gamma * R
#             batch_return.append(R)
#         batch_return.reverse()
#         batch_return = torch.tensor(batch_return, dtype=torch.float)
#
#         return batch_return
#
#     def calc_loss(self, done):
#         states = torch.tensor(self.states, dtype=torch.float)
#         actions = torch.tensor(self.actions, dtype=torch.float)
#
#         returns = self.calc_R(done)
#
#         # pi, values = self.forward(states)
#         pi = self.pi_net(states)
#         values = self.v_net(states)
#
#         values = values.squeeze()
#         critic_loss = (returns - values) ** 2
#
#         probs = torch.softmax(pi, dim=1)
#         dist = Categorical(probs)
#         log_probs = dist.log_prob(actions)
#         actor_loss = -log_probs * (returns - values)
#
#         total_loss = (critic_loss + actor_loss).mean()
#
#         return total_loss
#
#     def choose_action(self, observation):
#         state = torch.tensor([observation], dtype=torch.float)
#         # pi, v = self.forward(state)
#         pi = self.pi_net(state)
#         # v = self.v_net(state)
#         probs = torch.softmax(pi, dim=1)
#         dist = Categorical(probs)
#         action = dist.sample().numpy()[0]
#
#         return action


# class Agent(mp.Process):
#     def __init__(self, global_actor_critic, optimizer, scheduler, input_dims, n_actions,
#                  name, global_ep_idx, glob_episode_thred, global_dict, config):
#         super(Agent, self).__init__()
#         self.env = HyperGameSim()
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         print("Local Using", self.device)
#         self.local_actor_critic = ActorCritic(input_dims, n_actions,
#                                               gamma=config["gamma"], pi_net_struc=config["pi_net_struc"],
#                                               v_net_struct=config["v_net_struct"]).to(self.device)
#         self.global_actor_critic = global_actor_critic
#         self.name = 'w%02i' % name
#         print("creating: " + self.name)
#         self.episode_idx = global_ep_idx
#         self.optimizer = optimizer
#         self.scheduler = scheduler
#         self.glob_episode_thred = glob_episode_thred
#         self.shared_dict = global_dict
#         self.config = config
#
#         # self.writer = writer
#
#     def run(self):
#         # create writer for TensorBoard
#         # run 'tensorboard --logdir=runs' in terminal to start TensorBoard.
#         writer = SummaryWriter("runs_att/each_run_" + self.shared_dict["start_time"])
#
#         while self.episode_idx.value < self.glob_episode_thred:
#             # Episode start
#             t_step = 1
#             done = False
#             observation = self.env.reset()
#             score = 0
#             self.local_actor_critic.clear_memory()
#             while not done:
#                 action = self.local_actor_critic.choose_action(observation)
#                 self.shared_dict["att_action"][action] += 1
#                 observation_, _, reward_att, done, info = self.env.step(action_att=action)
#                 score += reward_att
#                 self.local_actor_critic.remember(observation, action, reward_att)
#                 if t_step % 5 == 0 or done:
#                     loss = self.local_actor_critic.calc_loss(done)
#                     self.optimizer.zero_grad()
#                     loss.backward()
#                     for local_param, global_param in zip(
#                             self.local_actor_critic.parameters(),
#                             self.global_actor_critic.parameters()):
#                         global_param._grad = local_param.grad
#                     self.optimizer.step()
#                     self.local_actor_critic.load_state_dict(
#                         self.global_actor_critic.state_dict())
#                     self.local_actor_critic.clear_memory()
#                 t_step += 1
#                 observation = observation_
#
#             with self.episode_idx.get_lock():
#                 self.episode_idx.value += 1
#
#             # this one commented out for save memory, uncomment as needed.
#             # save data
#             self.shared_dict["reward"].append(score)
#             # self.shared_dict["t_step"].append(t_step)       # number of round for a game
#
#             # self.scheduler.step()  # update learning rate each episode
#
#             print(self.name, 'global-episode ', self.episode_idx.value, 'reward %.1f' % score)
#             writer.add_scalar("Score", score, self.episode_idx.value)
#
#
#         global_reward = [ele for ele in self.shared_dict["reward"]]  # the return of the last 10 percent episodes
#         len_last_return = max(1, int(len(global_reward) * 0.1))     # max can make sure at lead one element in list
#         last_ten_percent_return = global_reward[-len_last_return:]
#
#         ave_10_per_return = sum(last_ten_percent_return) / len(last_ten_percent_return)
#
#         self.shared_dict["ave_10_per_return"].append(ave_10_per_return)
#
#         writer.flush()
#         writer.close()  # close SummaryWriter of TensorBoard
#         return


def objective(trial):
    start_time = time.time()
    if trial is not None:
        trial_num_str = str(trial.number)
    else:
        trial_num_str = "None"
    writer_hparam = SummaryWriter("runs/each_run_" +str(start_time) + "-Trial_" + trial_num_str)

    config = dict(glob_episode_thred=1200.0, gamma=0.3, lr=0.00020036, LR_decay=0.972, pi_net_struc=[384, 384, 512], v_net_struct=[256])    # this config may be changed by optuna

    # 2. Suggest values of the hyperparameters using a trial object.
    if trial is not None:
        config["glob_episode_thred"] = trial.suggest_int('glob_episode_thred', 1000, 1500, 100)     # total number of episodes
        # config["glob_episode_thred"] = trial.suggest_int('glob_episode_thred', 5, 5)  # total number of episodes
        config["gamma"] = trial.suggest_loguniform('gamma', 0.1, 1.0)
        config["lr"] = trial.suggest_loguniform('lr', 1e-7, 1e-1)
        config["LR_decay"] = trial.suggest_loguniform('LR_decay', 0.8, 1.0)   # since scheduler is not use. This one has no impact to reward
        pi_n_layers = trial.suggest_int('pi_n_layers', 3, 5)  # total number of layer
        config["pi_net_struc"] = []     # Reset before append
        for i in range(pi_n_layers):
            config["pi_net_struc"].append(trial.suggest_int(f'pi_n_units_l{i}', 32, 128, 32))   # try various nodes each layer
        v_n_layers = trial.suggest_int('v_n_layers', 3, 5)  # total number of layer
        config["v_net_struct"] = []     # Reset before append
        for i in range(v_n_layers):
            config["v_net_struct"].append(trial.suggest_int(f'v_n_units_l{i}', 32, 128, 32))  # try various nodes each layer
    print("config", config)

    if on_server:
        num_worker = 125  # mp.cpu_count()     # update this for matching server's resources
    else:
        num_worker = 2

    temp_env = HyperGameSim()
    n_actions = temp_env.action_space.n
    input_dims = temp_env.observation_space.shape
    temp_env.close_env()    # close client for avoiding client limit error

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Share Using", device)
    global_actor_critic = ActorCritic(input_dims, n_actions,
                                      gamma=config["gamma"],
                                      pi_net_struc=config["pi_net_struc"],
                                      v_net_struct=config["v_net_struct"]).to(device)  # global NN
    print(global_actor_critic)
    global_actor_critic.share_memory()
    optim = SharedAdam(global_actor_critic.parameters(), lr=config['lr'],
                       betas=(0.9, 0.999))
    global_ep = mp.Value('i', 0)

    def lambda_function(epoch):  # epoch increase one when scheduler.step() is called
        return config["LR_decay"] ** epoch

    # scheduler = LambdaLR(optim, lr_lambda=lambda_function)
    scheduler = None    # don't use scheduler



    shared_dict = {}
    shared_dict["reward"] = Manager().list()  # use Manager().list() to create a shared list between processes
    shared_dict["t_step"] = Manager().list()
    shared_dict["att_action"] = mp.Array('i', n_actions)
    shared_dict["start_time"] = str(start_time)
    shared_dict["ave_10_per_return"] = Manager().list()
    # shared_dict["reward"] = mp.Array('d', glob_episode_thred + num_worker)        # save simulation data ('d' means double-type)

    workers = [Agent(global_actor_critic,
                     optim,
                     scheduler,
                     input_dims,
                     n_actions,
                     name=i,
                     global_ep_idx=global_ep,
                     global_dict=shared_dict,
                     config=config,
                     glob_episode_thred=config['glob_episode_thred'],
                     player="att") for i in range(num_worker)]
    [w.start() for w in workers]
    [w.join() for w in workers]

    print("--- Simulation Time: %s seconds ---" % round(time.time() - start_time, 1))

    global_reward_10_per = [ele for ele in shared_dict["ave_10_per_return"]]    # get reward of all local agents
    ave_global_reward_10_per = sum(global_reward_10_per)/len(global_reward_10_per)

    # ========= Save global model =========
    if on_server:
        path = "/home/zelin/Drone/code_files/data/attacker"
    else:
        path = "/Users/wanzelin/办公/gym-drones/data/A3C/defender"
    os.makedirs(path + "/model", exist_ok=True)
    torch.save(global_actor_critic, path + "/model/trained_A3C_attacker_" + str(start_time))

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
    writer_hparam.add_hparams(temp_config, {'return_reward': ave_global_reward_10_per})  # add for Hyperparameter Tuning
    writer_hparam.flush()
    writer_hparam.close()

    return ave_global_reward_10_per  # return average value


if __name__ == '__main__':
    if platform == "darwin":
        on_server = False
    else:
        on_server = True
    # objective(None)
    # 3. Create a study object and optimize the objective function.
    # /home/zelin/Drone/data
    if on_server:
        study = optuna.create_study(direction='maximize', study_name="A3C-hyperparameter-study",
                                    storage="sqlite://////home/zelin/Drone/code_files/data/attacker/HyperPara_database.db",
                                    load_if_exists=True)
    else:
        study = optuna.create_study(direction='maximize', study_name="A3C-hyperparameter-study",
                                    storage="sqlite://///Users/wanzelin/办公/gym-drones/data/attacker/HyperPara_database.db",
                                    load_if_exists=True)
    study.optimize(objective, n_trials=100)

    # Saving data to file
    # Reward
    # global_reward = [ele for ele in shared_dict["reward"]]  # convert mp.Array to python list
    # print(f"reward {global_reward}")
    # os.makedirs("data/A3C", exist_ok=True)
    # the_file = open("data/A3C/reward_train_all_result.pkl",
    #                 "wb+")
    # pickle.dump(global_reward, the_file)
    # the_file.close()
    #
    # # t_step (number of round in a game)
    # global_t_step = [ele for ele in shared_dict["t_step"]]
    # print(f"t_step {global_t_step}")
    # os.makedirs("data/A3C", exist_ok=True)
    # the_file = open("data/A3C/t_step_all_result.pkl",
    #                 "wb+")
    # pickle.dump(global_t_step, the_file)
    # the_file.close()
    #
    # # defender action frequency
    # global_def_action = [ele for ele in shared_dict["def_action"]]
    # print(f"def_action {global_def_action}")
    # os.makedirs("data/A3C", exist_ok=True)
    # the_file = open("data/A3C/def_action_all_result.pkl",
    #                 "wb+")
    # pickle.dump(global_def_action, the_file)
    # the_file.close()
