'''
Project     ：gym-drones
File        ：A3C_model.py
Author      ：Zelin Wan
Date        ：8/15/22
Description : This is the model of A3C.
'''

import os
os.environ["OMP_NUM_THREADS"] = "1" # Error #34: System unable to allocate necessary resources for OMP thread:"
import torch as torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import optuna
import numpy as np
import gym
import copy

from torch.distributions import Categorical
# from Gym_HoneyDrone_Defender_and_Attacker import HyperGameSim
from multiprocessing import Manager
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR
from sys import platform


class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3):
        super(SharedAdam, self).__init__(params, lr=lr)


        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions, lr=0.01, LR_decay=0.99, gamma=0.99, epsilon = 0.1, epsilon_decay = 0.99,
                 pi_net_struc=None, v_net_struct=None, trial=None, fixed_seed=True):
        super(ActorCritic, self).__init__()

        self.input_dims = input_dims
        self.n_actions = n_actions
        self.lr = lr
        self.LR_decay = LR_decay
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        if pi_net_struc is None:
            pi_net_struc = [64, 64]     # default 64, 64 hidden layers if not specify
        else:
            self.pi_net_struc = pi_net_struc
        if v_net_struct is None:
            v_net_struct = [64, 64]     # default 64, 64 hidden layers if not specify
        else:
            self.v_net_struct = v_net_struct
        self.trial = trial      # save this for optuna pruning
        self.fixed_seed = fixed_seed        # if true, fixed seed will be used when initialize the model
        # if self.fixed_seed:
        #     torch.manual_seed(0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Local Using", self.device)
        self.pi_net = self.build_Net(*input_dims, n_actions, pi_net_struc).to(self.device)
        self.v_net = self.build_Net(*input_dims, 1, v_net_struct).to(self.device)
        self.pi_net.apply(self.init_weight_bias)  # initial weight (normal distribution)
        self.v_net.apply(self.init_weight_bias)  # initial weight (normal distribution)

        # self.pi1 = nn.Linear(*input_dims, 128)
        # self.v1 = nn.Linear(*input_dims, 128)
        # self.pi2 = nn.Linear(128, 256)
        # self.v2 = nn.Linear(128, 256)
        # self.pi3 = nn.Linear(256, 128)
        # self.v3 = nn.Linear(256, 128)
        # self.pi = nn.Linear(128, n_actions)
        # self.v = nn.Linear(128, 1)
        self.t_max = 5      # t_max as mentioned in A3C paper
        self.rewards = []
        self.actions = []
        self.states = []
        self.global_ep = mp.Value('i', 0)
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.optimizer = SharedAdam(self.parameters(), lr=self.lr)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.lambda_function)
        self.pruning = False

    def epsilon_decay_step(self):
        new_epsilon = self.epsilon * self.epsilon_decay
        # print("new new_epsilon", new_epsilon)
        self.epsilon = new_epsilon

    def lambda_function(self, epoch):  # epoch increase one when scheduler.step() is called
        return self.LR_decay ** epoch

    def init_weight_bias(self, layer):
        if type(layer) == nn.Linear:
            nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain('relu'))  # use normal distribution
            nn.init.normal_(layer.bias, std=1 / layer.in_features)

    def build_Net(self, obser_space, action_space, net_struc):
        layers = []
        in_features = obser_space
        # for i in range(n_layers):
        for node_num in net_struc:
            layers.append(nn.Linear(in_features, node_num))
            layers.append(nn.ReLU())
            in_features = node_num
        layers.append(nn.Linear(in_features, action_space))
        net = nn.Sequential(*layers)
        return net

    def remember(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []

    # def forward(self, state):
    #     pi1 = F.relu(self.pi1(state))
    #     v1 = F.relu(self.v1(state))
    #
    #     pi2 = self.pi2(pi1)
    #     v2 = self.v2(v1)
    #
    #     pi3 = F.relu(self.pi3(pi2))
    #     v3 = F.relu(self.v3(v2))
    #
    #     pi = self.pi(pi3)
    #     v = self.v(v3)
    #
    #     return pi, v

    # def calc_R(self, done):
    #     '''
    #
    #     Args:
    #         done: game end or not
    #
    #     Returns: the return with size t_max. This is the target for value network to calculate loss.
    #
    #     '''
    #     states = torch.tensor(self.states, dtype=torch.float)
    #     # _, v = self.forward(states)
    #     v = self.v_net(states)
    #
    #     R = v[-1] * (1 - int(done))
    #
    #     batch_return = []
    #     for reward in self.rewards[::-1]:
    #         R = reward + self.gamma * R
    #         batch_return.append(R)
    #     batch_return.reverse()
    #     batch_return = torch.tensor(batch_return, dtype=torch.float)
    #
    #     return batch_return

    def calc_R(self, done, observation_new):
        '''

        Args:
            done: game end or not

        Returns: the discounted return with size t_max. This is the target for value network to calculate loss.

        '''
        states = torch.tensor(self.states, dtype=torch.float)
        observation_new = torch.tensor(observation_new, dtype=torch.float)
        # _, v = self.forward(states)
        # v = self.v_net(states)
        v_next = self.v_net(observation_new)    # v for the state_{t_max+1}

        R = v_next * (1 - int(done))

        batch_return = []
        for reward in self.rewards[::-1]:
            R = reward + self.gamma * R
            batch_return.append(R)
        batch_return.reverse()
        batch_return = torch.tensor(batch_return, dtype=torch.float)

        return batch_return

    def calc_loss(self, done, observation_new):
        states = torch.tensor(self.states, dtype=torch.float)
        actions = torch.tensor(self.actions, dtype=torch.float)

        returns = self.calc_R(done, observation_new)     # TODO: find out what is this

        pi = self.pi_net(states)
        probs = torch.softmax(pi, dim=1)

        values = self.v_net(states)

        values = values.squeeze()
        critic_loss = (returns - values) ** 2

        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        actor_loss = -log_probs * (returns - values)

        total_loss = (critic_loss + actor_loss).mean()
        # print("total_loss", total_loss)
        # print("critic_loss", critic_loss)
        # print("actor_loss", actor_loss)
        return total_loss, critic_loss.mean(), actor_loss.mean()

    def choose_action(self, observation):

        if torch.rand((1,)).item() < self.epsilon:
            # random action
            print("Random action. Epsilon is", self.epsilon)
            return torch.randint(0, self.n_actions, (1,)).item()

        state = torch.tensor([observation], dtype=torch.float)
        # pi, v = self.forward(state)
        pi = self.pi_net(state)
        # v = self.v_net(state)
        probs = torch.softmax(pi, dim=1)
        # print("probs", probs)
        dist = Categorical(probs)
        action = dist.sample().numpy()[0]

        return action


class Agent(mp.Process):
    def __init__(self, global_actor_critic, input_dims, n_actions,
                 name, glob_episode_thred, global_dict, config, player="player"):
        super(Agent, self).__init__()
        # self.env = HyperGameSim()
        self.env = gym.make('CartPole-v1')
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print("Local Using", self.device)
        self.global_actor_critic = global_actor_critic
        self.local_actor_critic = ActorCritic(input_dims, n_actions,
                                              gamma=global_actor_critic.gamma, pi_net_struc=global_actor_critic.pi_net_struc,
                                              v_net_struct=global_actor_critic.v_net_struct)
        self.local_actor_critic.load_state_dict(self.global_actor_critic.state_dict())
        # print("local dict", self.local_actor_critic.state_dict())
        # self.local_actor_critic = copy.deepcopy(global_actor_critic)

        self.name_id = name
        self.name = 'w%02i' % name
        self.episode_idx = global_actor_critic.global_ep
        self.optimizer = global_actor_critic.optimizer
        self.scheduler = global_actor_critic.scheduler
        self.glob_episode_thred = glob_episode_thred
        self.shared_dict = global_dict
        self.config = config
        self.player = player    # "player" means not configured, "att" means attacker, "def" means defender

    def get_last_ten_ave(self, list):
        len_last_return = max(1, int(len(list) * 0.1))  # max can make sure at lead one element in list
        last_ten_percent_return = copy.copy(self.shared_dict["reward"][-len_last_return:])
        if len(last_ten_percent_return):
            ave_10_per_return = sum(last_ten_percent_return) / len(last_ten_percent_return)
        else:
            ave_10_per_return = 0

        return ave_10_per_return

    def run(self):
        # create writer for TensorBoard
        # run 'tensorboard --logdir=runs' in terminal to start TensorBoard.
        if self.name_id == 0:
            if self.global_actor_critic.trial is not None:
                trial_num_str = str(self.global_actor_critic.trial.number)
            else:
                trial_num_str = "None"

            if self.shared_dict["on_server"]:
                path = "/home/zelin/Drone/code_files/data/"
            else:
                path = ""
            writer = SummaryWriter(path + "runs_" + self.player + "/each_run_" + self.shared_dict["start_time"] + "-" +
                                   self.player + "-" + "-Trial_" + trial_num_str + "-eps")
            # print("creating writer", "runs_"+self.player+"/each_run_" + self.shared_dict["start_time"] + "-" + self.player + "-" + "-Trial_" + trial_num_str + "-eps")

        else:
            writer = None

        # fix seed for the whole trainning
        # self.env.set_random_seed()
        total_loss_set = []
        critic_loss_set = []
        actor_loss_set = []

        while self.episode_idx.value < self.glob_episode_thred:
            # Episode start
            # self.env.render(mode = "human")
            t_step = 1
            done = False
            observation = self.env.reset()
            # if self.player == "att":
            #     the_observation = observation['att']
            # elif self.player == "def":
            #     the_observation = observation['def']
            # else:
            #     print("Error: player is not specified, using defender's observation")
            #     the_observation = observation['def']
            the_observation = observation       # TODO: this is a test
            score = 0
            oppo_score = 0
            self.local_actor_critic.clear_memory()
            while not done:
                action = self.local_actor_critic.choose_action(the_observation)
                self.shared_dict["action"][action] += 1
                action_def = None
                action_att = None

                # determine wether it's defender's action or attacker's action
                if self.player == "att":
                    action_att = action
                elif self.player == "def":
                    action_def = action
                else:
                    print("Error: player is not specified, using action for defender")
                    action_def = action

                # observation_new, reward, done, info = self.env.step(action_def=action_def, action_att=action_att)

                # if self.player == "att":
                #     the_reward = reward['att']
                #     oppo_reward = reward['def']
                #     the_observation_new = observation_new['att']
                # elif self.player == "def":
                #     the_reward = reward['def']
                #     oppo_reward = reward['att']
                #     the_observation_new = observation_new['def']
                # else:
                #     print("Error: player is not specified, using defender's reward")
                #     the_reward = reward['def']
                #     oppo_reward = reward['att']
                #     the_observation_new = observation_new['def']

                observation_new, reward, done, info = self.env.step(action) # TODO: this is test

                the_observation_new = observation_new  # TODO: this is test
                the_reward = reward
                oppo_reward = 0

                score += the_reward         # current player's score
                oppo_score += oppo_reward   # opponent's score

                # memory
                self.local_actor_critic.remember(the_observation, action, the_reward)
                if t_step % self.local_actor_critic.t_max == 0 or done:
                    total_loss, critic_loss, actor_loss = self.local_actor_critic.calc_loss(done, the_observation_new)

                    total_loss_set.append(total_loss.item())
                    critic_loss_set.append(critic_loss.item())
                    actor_loss_set.append(actor_loss.item())

                    self.optimizer.zero_grad()
                    total_loss.backward()
                    for local_param, global_param in zip(self.local_actor_critic.parameters(),
                                                         self.global_actor_critic.parameters()):
                        global_param._grad = local_param.grad
                    self.optimizer.step()
                    # copy global parameter to local parameter
                    self.local_actor_critic.load_state_dict(self.global_actor_critic.state_dict())
                    # copy global epsilon to local epsilon
                    self.local_actor_critic.epsilon = self.global_actor_critic.epsilon
                    # clear memory
                    self.local_actor_critic.clear_memory()

                t_step += 1
                the_observation = the_observation_new


            with self.episode_idx.get_lock():
                self.episode_idx.value += 1
                self.scheduler.step()  # update learning rate each episode
                self.global_actor_critic.epsilon_decay_step()   # update epsilon each episode

                # save data to shared dictionary for tensorboard
                self.shared_dict["score"].append(score)
                self.shared_dict["oppo_score"].append(oppo_score)
                self.shared_dict["lr"].append(self.optimizer.param_groups[0]['lr'])
                self.shared_dict["epsilon"].append(self.global_actor_critic.epsilon)
                self.shared_dict["eps"].append(self.episode_idx.value)

            # tensorboard writer. Only agent (index 0) can write to tensorboard
            if writer is not None:
                # print("writing", score, self.episode_idx.value)
                for id in range(self.shared_dict["index"], len(self.shared_dict["eps"])):
                    # write score
                    writer.add_scalar("Average Score", self.shared_dict["score"][id], self.shared_dict["eps"][id])
                    writer.add_scalar("Opponent's Average Score", self.shared_dict["oppo_score"][id], self.shared_dict["eps"][id])
                    # write lr
                    writer.add_scalar("Learning rate", self.shared_dict["lr"][id], self.shared_dict["eps"][id])
                    # write epsilon
                    writer.add_scalar("Epsilon (random action probability)", self.shared_dict["epsilon"][id], self.shared_dict["eps"][id])
                    # write mission time (step)
                    writer.add_scalar("Mission Time (step)", t_step, self.shared_dict["eps"][id])
                    # write loss
                    writer.add_scalar("Model's Total Loss", sum(total_loss_set) / len(total_loss_set),
                                      self.shared_dict["eps"][id])
                    writer.add_scalar("Model's Critic Loss", sum(critic_loss_set) / len(critic_loss_set),
                                      self.shared_dict["eps"][id])
                    writer.add_scalar("Model's Actor Loss", sum(actor_loss_set) / len(actor_loss_set),
                                      self.shared_dict["eps"][id])
                    # mission completion rate
                    # writer.add_scalar("Mission Success Rate (completion rate)", self.env.system.scanCompletePercent(), self.shared_dict["eps"][id])   # TODO: comment out for test
                # update index
                self.shared_dict["index"] = len(self.shared_dict["eps"])

            # this one commented out for save memory, uncomment as needed.
            # save data
            self.shared_dict["reward"].append(score)
            # self.shared_dict["t_step"].append(t_step)       # number of round for a game

            # Pruning:
            # Report intermediate objective value.
            min_episode = self.config["min_episode"]  # allow each trial run at least this episodes
            if self.global_actor_critic.trial is not None:
                ave_10_per_return = self.get_last_ten_ave(self.shared_dict["reward"])
                # len_last_return = max(1, int(len(self.shared_dict["reward"]) * 0.1))  # max can make sure at lead one element in list
                # last_ten_percent_return = copy.copy(self.shared_dict["reward"][-len_last_return:])
                # if len(last_ten_percent_return):
                #     ave_10_per_return = sum(last_ten_percent_return) / len(last_ten_percent_return)
                # else:
                #     ave_10_per_return = 0
                self.global_actor_critic.trial.report(ave_10_per_return, self.episode_idx.value)
                # Handle pruning based on the intermediate value.
                if self.episode_idx.value > min_episode and self.global_actor_critic.trial.should_prune():  # after 'min_episode', if reward is lower than the average of other trials' reward, terminate this trial.
                    print("Mark this trial as pruning")
                    self.global_actor_critic.pruning = True     # mark pruning as True


                # only agent (index 0) can raise optuna's pruning
                if self.global_actor_critic.pruning:
                    if self.name_id == 0:
                        print("Pruning This Trial!!!!!!!!")
                        raise optuna.TrialPruned()

            print(self.name, 'global-episode ', self.episode_idx.value, 'reward %.1f' % score)




        global_reward = [ele for ele in self.shared_dict["reward"]]  # the return of the last 10 percent episodes
        len_last_return = max(1, int(len(global_reward) * 0.1))     # max can make sure at lead one element in list
        last_ten_percent_return = global_reward[-len_last_return:]

        if len(last_ten_percent_return):
            ave_10_per_return = sum(last_ten_percent_return) / len(last_ten_percent_return)
        else:
            ave_10_per_return = 0

        self.shared_dict["ave_10_per_return"].append(ave_10_per_return)

        if writer is not None:
            writer.flush()
            writer.close()  # close SummaryWriter of TensorBoard
        return