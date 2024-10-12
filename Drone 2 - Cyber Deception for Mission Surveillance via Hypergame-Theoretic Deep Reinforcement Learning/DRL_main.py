'''
Project     ：Drone-DRL-HT 
File        ：DRL_main.py
Author      ：Zelin Wan
Date        ：2/7/23
Description : 
'''


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
from Gym_Defender_and_Attacker import HyperGameSim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR


class DQN(nn.Module):
    def __init__(self, obser_space, action_space,
                 batch_size=32,
                 learning_rate=0.1, lr_decay_def=0.999,
                 epsilon=1.0, epsilon_decay=0.99,
                 gamma=0.9,
                 target_replace_iter=100,
                 memory_size=1000):
        super(DQN, self).__init__()
        self.eval_net = self.build_Net(obser_space, action_space)
        self.target_net = self.build_Net(obser_space, action_space)
        # initialize parameters (weight and bias)
        self.eval_net.apply(self.init_weight_bias)
        self.target_net.load_state_dict(self.eval_net.state_dict())

        self.dim_state = obser_space  # 状态维度
        self.n_actions = action_space  # 可选动作数
        self.batch_size = batch_size  # 小批量梯度下降，每个“批”的size
        self.learning_rate = learning_rate  # 学习率
        self.lr_decay_def = lr_decay_def
        self.epsilon = epsilon  # probability of NOT randomly selection pybullet_action. 贪婪系数
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma  # 回报衰减率
        self.memory_size = memory_size  # 记忆库的规格
        self.target_replace_iter = target_replace_iter  # update target_net every this number (of time). target网络延迟更新的间隔步数
        self.learn_step_counter = 0  # count the number of time passed. 在计算隔n步跟新的的时候用到
        self.memory_counter = 0  # 用来计算存储索引
        # each horizontal line contain: [previous state, action, reward, new state]
        self.memory = np.zeros((self.memory_size, self.dim_state * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.learning_rate)    # optimizer
        self.loss_func = nn.MSELoss()       # loss function
        self.scheduler_def = LambdaLR(self.optimizer, lr_lambda=self.lambda_function_def)


    def build_Net(self, obser_space, action_space):
        net = nn.Sequential(
            nn.Linear(obser_space, 50),
            nn.ReLU(),
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Linear(25, action_space)
        )
        return net

    def init_weight_bias(self, layer):
        if type(layer) == nn.Linear:
            nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain('tanh'))  # use normal distribution
            nn.init.constant_(layer.bias, 0.)

    def lambda_function_def(self, epoch):  # epoch increase one when scheduler_def.step() is called
        return self.lr_decay_def ** epoch

    def epsilon_decay_step(self):
        new_epsilon = self.epsilon * self.epsilon_decay
        self.epsilon = new_epsilon

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
            print("renew target network")
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

        return loss

    # save one state to memory
    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))

        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1




def train_DRL():
    # create writer for TensorBoard
    # run 'tensorboard --logdir=runs' in terminal to start TensorBoard.
    writer = SummaryWriter()

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = DQN().to(device)
    # print_debug(model)
    system = system_model()
    defender = defender_model(system)

    is_custom_env = True
    if is_custom_env:
        env = HyperGameSim(target_size=10, fixed_seed=False)
    else:
        env = gym.make('CartPole-v1')

    # DQN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using", device)

    if is_custom_env:
        model = DQN(env.observation_space['def'].shape[0], env.action_space['def'].n).to(device)
    else:
        model = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
    print(model)

    episodes = 200
    loss_def = 0
    for episode in range(episodes):
        obs_out = env.reset()
        if is_custom_env:
            obs_def = obs_out['def']
        else:
            obs_def = obs_out[0]
        done = False
        score_def = 0
        info = {}
        step = 0
        while not done:
            step += 1
            action_def = model.choose_action(obs_def)      # select an pybullet_action
            # action_def = np.array(int(random.uniform(0, 2)))    # random action for testing

            if is_custom_env:
                obs_out_, reward, done, info = env.step(action_def)  # execute environment
                obs_def_ = obs_out_['def']
                reward_def = reward['def']
                print("reward_def", reward_def)
            else:
                obs_out_, reward, done, info, _ = env.step(action_def)  # execute environment
                obs_def_ = obs_out_
                reward_def = reward

            model.store_transition(obs_def, action_def, reward_def, obs_def_)     # save informatin to memory
            score_def += reward_def     # accmulate reward

            # train network when memory is full
            if model.memory_counter > model.memory_size:
                loss_def = model.learn()

            obs_def = obs_def_
            if not is_custom_env:
                env.render()

        model.scheduler_def.step()
        model.epsilon_decay_step()
        print('Episode:{} Score:{} epsilon:{}'.format(episode, score_def, model.epsilon))
        writer.add_scalar("Score", score_def, episode)
        writer.add_scalar("Loss", loss_def, episode)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)
    writer.flush()
    writer.close()  # close SummaryWriter of TensorBoard
    return





if __name__ == "__main__":

    # run simulation
    train_DRL()
