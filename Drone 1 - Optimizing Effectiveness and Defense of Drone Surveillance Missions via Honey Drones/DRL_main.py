import pickle

from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import os
import random
import gym
import math
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from model_System import system_model
from model_Defender import defender_model
from model_Attacker import attacker_model
from Gym_HoneyDrone_defender_only import HyperGameSim
from torch.utils.tensorboard import SummaryWriter

# class DroneSim(Env):
#     def __init__(self, defender):
#         self.defender = defender
#         self.action_space = Discrete(self.defender.strategy)
#         self.mission_time_max = self.defender.system.mission_max_duration
#         self.mission_time = 0       # state - 1
#         self.surv_complete = 0      # state - 2
#         # [time duration, ratio of mission complete]
#         self.observation_space = Box(low=np.array([0., 0.]), high=np.array([self.defender.system.mission_max_duration, 1.]))
#
#     def step(self, action):
#         # self.defender.set_strategy(pybullet_action)
#         # self.defender.pybullet_action()
#
#         self.mission_time += 1
#
#         state = [self.mission_time, self.surv_complete]
#         reward = 0
#         if self.mission_time < 2: #self.mission_time_max:
#             done = False
#         else:
#             done = True
#         info = {}
#         return state, reward, done, info
#
#
#     def render(self, *args):
#         pass
#
#     def reset(self, *args):
#         self.mission_time = 0
#         self.surv_complete = 0
#         return [self.mission_time, self.surv_complete]

# TODO: adjust more hyperparameters
# TODO: save trained defender's model


class DQN(nn.Module):
    def __init__(self, obser_space, action_space,
                 batch_size=32,
                 learning_rate=0.01,
                 epsilon=0.9,
                 gamma=0.9,
                 target_replace_iter=100,
                 memory_size=100):
        super(DQN, self).__init__()
        self.eval_net = self.build_Net(obser_space, action_space)
        self.target_net = self.build_Net(obser_space, action_space)

        self.dim_state = obser_space  # 状态维度
        self.n_actions = action_space  # 可选动作数
        self.batch_size = batch_size  # 小批量梯度下降，每个“批”的size
        self.learning_rate = learning_rate  # 学习率
        self.epsilon = epsilon  # probability of NOT randomly selection pybullet_action. 贪婪系数
        self.gamma = gamma  # 回报衰减率
        self.memory_size = memory_size  # 记忆库的规格
        self.target_replace_iter = target_replace_iter  # update target_net every this number (of time). target网络延迟更新的间隔步数
        self.learn_step_counter = 0  # count the number of time passed. 在计算隔n步跟新的的时候用到
        self.memory_counter = 0  # 用来计算存储索引
        # each horizontal line contain: [previous state, action, reward, new state]
        self.memory = np.zeros((self.memory_size, self.dim_state * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.learning_rate)    # optimizer
        self.loss_func = nn.MSELoss()       # loss function


    def build_Net(self, obser_space, action_space):
        net = nn.Sequential(
            nn.Linear(obser_space, 50),
            nn.ReLU(),
            nn.Linear(50, action_space)
        )
        return net

    # def forward(self, x):
    #     return self.net(x)

    def choose_action(self, x):
        X = torch.unsqueeze(torch.FloatTensor(x), 0)    # transfer state to tensor
        if np.random.uniform() < self.epsilon:
            action_value = self.eval_net.forward(X)              # forward in network
            action = torch.max(action_value, 1)[1]          # get pybullet_action with max value
            action = int(action)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    # back-propagattion
    def learn(self):
        # update target_net parameter every 'target_replace_iter' time
        if self.learn_step_counter % self.target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # get data from memory
        data_size = self.memory_size if self.memory_counter > self.memory_size else self.memory_counter

        sample_index = np.random.choice(data_size, self.batch_size)
        b_memory = self.memory[sample_index, :]
        # [0 : b_s : b_a : b_r: b_s_]
        b_s = torch.FloatTensor(b_memory[:, :self.dim_state])       # previous state
        b_a = torch.LongTensor(b_memory[:, self.dim_state:self.dim_state + 1].astype(int))  # integer used for torch.gather dimension
        b_r = torch.FloatTensor(b_memory[:, self.dim_state + 1:self.dim_state + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -self.dim_state:])     # next state

        # calculate LOSS
        q_eval = self.eval_net(b_s).gather(1, b_a)  # get result for previous state
        # print("self.eval_net(b_s)", self.eval_net(b_s))
        # print("b_a", b_a)
        # print("q_eval", q_eval)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
        # print("q_target", q_target)

        loss = self.loss_func(q_eval, q_target)
        # print("loss", loss)
        # quit()
        # back propagation
        self.optimizer.zero_grad()  # since gradient value is accumulated, reset it before use.
        loss.backward()
        self.optimizer.step()

    # save one state to memory
    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))

        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1




def train_DRL():
    # test environment
    # env = HyperGameSim()
    # glob_episode_thred = 1
    # action = env.action_space.sample()
    # print_debug(f"action: {action}")
    # env.step(action)
    # for episode in range(1, glob_episode_thred + 1):
    #     state = env.reset()
    #     done = False
    #     score = 0
    #
    #     while not done:
    #         # env.render()
    #         action = env.action_space.sample()
    #         n_state, reward, done, info = env.step(action)
    #         score += reward
    #     print_debug('Episode:{} Score:{}'.format(episode, score))
    # quit()

    # create writer for TensorBoard
    # run 'tensorboard --logdir=runs' in terminal to start TensorBoard.
    writer = SummaryWriter()

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = DQN().to(device)
    # print_debug(model)
    system = system_model()
    defender = defender_model(system)
    env = HyperGameSim()
    print("pybullet_action space", env.action_space.sample())
    print("observation space", env.observation_space.sample())

    # DQN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using", device)
    model = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
    print(model)

    episodes = 2
    for episode in range(episodes):
        state_pre = env.reset()
        done = False
        score = 0
        info = {}
        while not done:
            # env.render()
            a = model.choose_action(state_pre)      # select an pybullet_action
            state_new, reward, done, info = env.step(a)     # execute pybullet_action
            model.store_transition(state_pre, a, reward, state_new)     # save informatin to memory
            score += reward     # accmulate reward

            # train network when memory is full
            if model.memory_counter > model.memory_size:
                model.learn()

            state_pre = state_new
            # save data
            defense_strategy_list.append(a)

        print('Episode:{} Score:{}'.format(episode, score))
        writer.add_scalar("Score", score, episode)
        reward_train_all_result.append(score)
        game_succ_condition.append(info["mission condition"])

    writer.flush()
    writer.close()  # close SummaryWriter of TensorBoard
    return





if __name__ == "__main__":
    reward_train_all_result = []  # reward list for saving to file
    game_succ_condition = []    # False means game failed at index episode
    defense_strategy_list = []     # a list of defense strategy taken each round

    # run simulation
    train_DRL()

    print("result:", reward_train_all_result, game_succ_condition)
    # save rewards to file
    # reward_train_all_result
    os.makedirs("data", exist_ok=True)
    the_file = open("data/reward_train_all_result.pkl",
                    "wb+")
    pickle.dump(reward_train_all_result, the_file)
    the_file.close()

    # game_succ_condition
    os.makedirs("data", exist_ok=True)
    the_file = open("data/game_succ_condition.pkl",
                    "wb+")
    pickle.dump(game_succ_condition, the_file)
    the_file.close()

    # defense_strategy_list
    os.makedirs("data", exist_ok=True)
    the_file = open("data/defense_strategy_list.pkl",
                    "wb+")
    pickle.dump(defense_strategy_list, the_file)
    the_file.close()

    # try:
    #     os.system('say "your program has finished"')
    # except:
    #     print("command not found: say")










