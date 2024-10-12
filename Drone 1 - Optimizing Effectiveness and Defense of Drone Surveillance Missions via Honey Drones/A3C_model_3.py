'''
Project     ：gym-drones 
File        ：A3C_model_3.py
Author      ：Zelin Wan
Date        ：9/3/22
Description : The agent that will be used by 'A3C_train_agent_optuna_3.py' for parallel learning.
'''

import os
os.environ["OMP_NUM_THREADS"] = "1" # Error #34: System unable to allocate necessary resources for OMP thread:"
import torch
import torch.nn as nn
from utils import v_wrap, set_init, push_and_pull, record
import torch.nn.functional as F
import torch.multiprocessing as mp
import gym
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from Gym_HoneyDrone_Defender_and_Attacker import HyperGameSim


class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions, epsilon=0.1, epsilon_decay=0.9, pi_net_struc=None, v_net_struct=None, fixed_seed=True):
        super(ActorCritic, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.epsilon = epsilon
        print("self.epsilon", self.epsilon)
        self.epsilon_decay = epsilon_decay
        if pi_net_struc is None:
            pi_net_struc = [128]     # default 64, 64 hidden layers if not specify
        else:
            self.pi_net_struc = pi_net_struc
        if v_net_struct is None:
            v_net_struct = [128]     # default 64, 64 hidden layers if not specify
        else:
            self.v_net_struct = v_net_struct

        if fixed_seed:
            print("fix init seed")
            torch.manual_seed(0)
        self.pi_net = self.build_Net(input_dims, n_actions, pi_net_struc)
        self.v_net = self.build_Net(input_dims, 1, v_net_struct)

        self.pi_net.apply(self.init_weight_bias)  # initial weight (normal distribution)
        self.v_net.apply(self.init_weight_bias)  # initial weight (normal distribution)

        # self.s_dim = input_dims
        # self.a_dim = n_actions
        # self.pi1 = nn.Linear(input_dims, 128)
        # self.pi2 = nn.Linear(128, n_actions)
        # self.v1 = nn.Linear(input_dims, 128)
        # self.v2 = nn.Linear(128, 1)
        # set_init([self.pi1, self.pi2, self.v1, self.v2], fixed_seed)

        self.distribution = torch.distributions.Categorical

    def epsilon_decay_step(self):
        new_epsilon = self.epsilon * self.epsilon_decay
        self.epsilon = new_epsilon

    def init_weight_bias(self, layer):
        if type(layer) == nn.Linear:
            nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain('tanh'))  # use normal distribution
            # nn.init.kaiming_normal_(layer.weight)
            # nn.init.normal_(layer.weight, mean=0., std=0.1)
            # nn.init.normal_(layer.bias, std=1 / layer.in_features)
            nn.init.constant_(layer.bias, 0.)

    def build_Net(self, obser_space, action_space, net_struc):
        layers = []
        in_features = obser_space
        # for i in range(n_layers):
        for node_num in net_struc:
            layers.append(nn.Linear(in_features, node_num))
            layers.append(nn.Tanh())
            in_features = node_num
        layers.append(nn.Linear(in_features, action_space))
        net = nn.Sequential(*layers)
        return net

    # def forward(self, x):
    #     pi1 = torch.tanh(self.pi1(x))
    #     logits = self.pi2(pi1)
    #     v1 = torch.tanh(self.v1(x))
    #     values = self.v2(v1)
    #     return logits, values
    # def forward(self, x):
    #     logits = self.pi_net
    #     values = self.v_net
    #     return logits, values

    def choose_action(self, obs):
        if torch.rand((1,)).item() < self.epsilon:
            # random action
            print("Random action. Epsilon is", self.epsilon)
            return np.int64(torch.randint(0, self.n_actions, (1,)).item())

        # self.eval()
        # logits, _ = self.forward(s)
        logits = self.pi_net(obs)
        prob = F.softmax(logits, dim=1).data
        # print("prob", prob)
        # if torch.sum(torch.isnan(prob)):    # avoid 'nan' error. When [nan], randomly choose an action
        #     return torch.randint(0, len(prob), (1,)).numpy()[0]
        # else:
        #     m = self.distribution(prob)
        #     return m.sample().numpy()[0]
        # print("Pi Net Prob:", prob)
        m = self.distribution(prob)
        return m.sample().numpy()[0]

    def loss_func(self, s, a, v_t):
        # print("v_t", v_t)
        self.train()
        # logits, values = self.forward(s)
        print("here")
        logits = self.pi_net(s)
        print("logits", logits)
        values = self.v_net(s)
        td = v_t - values
        c_loss = td.pow(2)

        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        # total_loss = (c_loss + a_loss).mean()
        c_loss_mean = c_loss.mean()
        a_loss_mean = a_loss.mean()
        total_loss = c_loss_mean + a_loss_mean
        # print(c_loss, a_loss, total_loss)
        return total_loss, c_loss_mean, a_loss_mean


class Agent(mp.Process):
    def __init__(self, gnet, opt, shared_dict, lr_decay, gamma, epsilon, epsilon_decay, MAX_EP, fixed_seed, trial, name_id: int, player='def', exist_model=False, is_custom_env=True):
        super(Agent, self).__init__()
        self.name_id = name_id
        self.name = 'w%02i' % name_id
        self.gnet = gnet
        self.opt = opt
        self.shared_dict = shared_dict
        self.g_ep = shared_dict['global_ep']
        self.g_r_list = shared_dict['glob_r_list']
        self.lr_decay = lr_decay
        self.exist_model = exist_model
        self.is_custom_env = is_custom_env
        if is_custom_env:
            self.env = HyperGameSim(fixed_seed=fixed_seed)
            self.N_S = self.env.observation_space[player].shape[0]
            self.N_A = self.env.action_space[player].n
        else:
            self.env = gym.make('CartPole-v1')
            self.N_S = self.env.observation_space.shape[0]
            # self.N_S = self.env.observation_space.n
            self.N_A = self.env.action_space.n
        self.lnet = ActorCritic(input_dims=self.N_S, n_actions=self.N_A, epsilon=epsilon, epsilon_decay=epsilon_decay, pi_net_struc=gnet.pi_net_struc,
                          v_net_struct=gnet.v_net_struct, fixed_seed=fixed_seed)  # local network
        # load pre-trained model's parameters
        if self.exist_model:
            path = "trained_model/" + player + "/trained_A3C"
            self.lnet.load_state_dict(torch.load(path))
            self.lnet.eval()
        else:
            self.lnet.train()

        self.gamma = gamma
        self.MAX_EP = MAX_EP
        self.trial = trial
        self.player = player
        self.scheduler = LambdaLR(self.opt, lr_lambda=self.lambda_function)

    def lambda_function(self, epoch):  # epoch increase one when scheduler.step() is called
        return self.lr_decay ** epoch

    def run(self):
        # ======== Create Writer for TensorBoard ========
        # run 'tensorboard --logdir=runs' in terminal to start TensorBoard.
        if self.name_id == 0:
            if self.trial is not None:
                trial_num_str = str(self.trial.number)
            else:
                trial_num_str = "None"

            if self.shared_dict["on_server"]:
                path = "/home/zelin/Drone/data/"
            else:
                path = ""
            writer = SummaryWriter(log_dir=path + "runs_" + self.player + "/each_run_" + self.shared_dict["start_time"] + "-" +
                                   self.player + "-" + "-Trial_" + trial_num_str + "-eps")
            # writer = None
            # print("creating writer", "runs_"+self.player+"/each_run_" + self.shared_dict["start_time"] + "-" + self.player + "-" + "-Trial_" + trial_num_str + "-eps")

        else:
            writer = None

        ep_counter = 0
        while self.g_ep.value < self.MAX_EP:
            with self.g_ep.get_lock():
                temp_ep = self.g_ep.value

            if self.is_custom_env:
                obs_out = self.env.reset()
                if self.player == "att":
                    obs = obs_out['att']
                elif self.player == "def":
                    obs = obs_out['def']
                else:
                    raise Exception("invalide 'player_name'")
            else:
                obs = self.env.reset()

            buffer_s, buffer_a, buffer_r = [], [], []
            total_loss_set = []
            c_loss_set = []
            a_loss_set = []
            att_succ_rate_set = []
            att_reward_0 = 0
            att_reward_1 = 0
            att_reward_2 = 0
            def_reward_0 = 0
            def_reward_1 = 0
            def_reward_2 = 0
            score_att = 0
            score_def = 0
            total_step = 1
            done = False
            while not done:
                # if self.name_id == 0:
                #     self.env.render()

                # choose action
                action = self.lnet.choose_action(v_wrap(obs[None, :]))
                if self.is_custom_env:      # for Drone environment
                    # assign action to attacker or defender
                    action_def = None
                    action_att = None
                    if self.player == "att":
                        action_att = action
                    elif self.player == "def":
                        action_def = action
                    else:
                        print("Error: player is not specified, using action for defender")
                        action_def = action
                    # interaction with environment
                    the_obs_, the_reward, done, info = self.env.step(action_def=action_def, action_att=action_att)
                    # the_obs_, the_reward, done, info = self.env.step()      # TODO: this is a test
                    # extract different reward and observation for attacker and defender
                    reward_att = the_reward['att']
                    reward_def = the_reward['def']
                    if self.player == "att":
                        obs_ = the_obs_['att']
                    elif self.player == "def":
                        obs_ = the_obs_['def']
                    else:
                        raise Exception("invalide 'player_name'")
                    # add data from each step (for custom env only)
                    # att_succ_rate_set.append(info['att_succ_rate'])
                    if info['att_counter']:     # this avoids dividing by zero
                        att_succ_rate_set.append(info['att_succ_counter']/info['att_counter'])
                    att_reward_0 += info["att_reward_0"]
                    att_reward_1 += info["att_reward_1"]
                    att_reward_2 += info["att_reward_2"]
                    def_reward_0 += info["def_reward_0"]
                    def_reward_1 += info["def_reward_1"]
                    def_reward_2 += info["def_reward_2"]
                else:
                    obs_, reward_def, done, _ = self.env.step(action)   # assign reward to reward_def, ignore reward_att
                    reward_att = 0

                # add data from each step
                score_att += reward_att
                score_def += reward_def

                if not self.exist_model:  # no need to train a trained model
                    # add to memory
                    buffer_a.append(action)
                    buffer_s.append(obs)
                    if self.player == "att":
                        buffer_r.append(reward_att)
                    elif self.player == "def":
                        buffer_r.append(reward_def)
                    else:
                        raise Exception("invalide 'player_name'")
                    # buffer_r.append(reward)

                    # update global and assign to local net
                    if total_step % 5 == 0 or done:
                        print("here")
                        # sync
                        total_loss, c_loss, a_loss = push_and_pull(self.opt, self.lnet, self.gnet, done, obs_, buffer_s,
                                                                   buffer_a, buffer_r, self.gamma)
                        # save loss data for display
                        total_loss_set.append(total_loss.item())
                        c_loss_set.append(c_loss.item())
                        a_loss_set.append(a_loss.item())
                        # clear memory
                        buffer_s, buffer_a, buffer_r = [], [], []

                obs = obs_
                total_step += 1

            score_def_avg = score_def / total_step
            score_att_avg = score_att / total_step
            # save data to shared dictionary for tensorboard
            with self.g_ep.get_lock():
                self.g_ep.value += 1
                self.shared_dict['eps_writer'].put(temp_ep)
                if self.player == "att":
                    self.g_r_list.append(score_att)
                else:
                    self.g_r_list.append(score_def)
                self.shared_dict["score_def_writer"].put(score_def)
                self.shared_dict["score_att_writer"].put(score_att)
                self.shared_dict["score_def_avg_writer"].put(score_def_avg)
                self.shared_dict["score_att_avg_writer"].put(score_att_avg)
                self.shared_dict["lr_writer"].put(self.opt.param_groups[0]['lr'])
                self.shared_dict["epsilon_writer"].put(self.lnet.epsilon)
                self.shared_dict['t_loss_writer'].put(sum(total_loss_set)/len(total_loss_set) if len(total_loss_set) else 0)
                self.shared_dict['c_loss_writer'].put(sum(c_loss_set) / len(c_loss_set) if len(c_loss_set) else 0)
                self.shared_dict['a_loss_writer'].put(sum(a_loss_set) / len(a_loss_set) if len(a_loss_set) else 0)
                if self.is_custom_env:
                    self.shared_dict['att_succ_rate_writer'].put(sum(att_succ_rate_set) / len(att_succ_rate_set) if len(att_succ_rate_set) else 0)
                    self.shared_dict['mission_condition_writer'].put(info["mission_condition"])
                    self.shared_dict['total_energy_consump_writer'].put(info["total_energy_consump"])
                    self.shared_dict['scan_percent_writer'].put(info["scan_percent"])
                    self.shared_dict['att_reward_0_writer'].put(att_reward_0)
                    self.shared_dict['att_reward_1_writer'].put(att_reward_1)
                    self.shared_dict['att_reward_2_writer'].put(att_reward_2)
                    self.shared_dict['def_reward_0_writer'].put(def_reward_0)
                    self.shared_dict['def_reward_1_writer'].put(def_reward_1)
                    self.shared_dict['def_reward_2_writer'].put(def_reward_2)


            # ==== tensorboard writer ====
            # Only agent (index 0) can write to tensorboard
            if writer is not None:
                while not self.shared_dict['eps_writer'].empty():
                    # use episode as index of tensorboard
                    current_eps = self.shared_dict['eps_writer'].get()
                    current_eps = ep_counter
                    # write score
                    writer.add_scalar("Accumulated Reward Defender", self.shared_dict["score_def_writer"].get(), current_eps)
                    writer.add_scalar("Accumulated Reward Attacker", self.shared_dict["score_att_writer"].get(), current_eps)
                    writer.add_scalar("Averaged Reward Defender", self.shared_dict["score_def_avg_writer"].get(), current_eps)
                    writer.add_scalar("Averaged Reward Attacker", self.shared_dict["score_att_avg_writer"].get(), current_eps)
                    # write lr
                    writer.add_scalar("Learning rate", self.shared_dict["lr_writer"].get(), current_eps)
                    # write epsilon
                    writer.add_scalar("Epsilon (random action probability)", self.shared_dict["epsilon_writer"].get(),
                                      current_eps)
                    # write mission time (step)
                    writer.add_scalar("Mission Time (step)", total_step, current_eps)
                    # write loss
                    writer.add_scalar("Model's Total Loss", self.shared_dict['t_loss_writer'].get(), current_eps)
                    writer.add_scalar("Model's Critic Loss", self.shared_dict['c_loss_writer'].get(), current_eps)
                    writer.add_scalar("Model's Actor Loss", self.shared_dict['a_loss_writer'].get(), current_eps)
                    # write mission completion rate
                    if self.is_custom_env:
                        # write attack success rate
                        writer.add_scalar("Attack Success Rate", self.shared_dict['att_succ_rate_writer'].get(),
                                          current_eps)
                        # write mission success rate
                        writer.add_scalar("Mission Success Rate (completion rate)", self.shared_dict['scan_percent_writer'].get(), current_eps)
                        # # write missin result
                        # writer.add_scalar("Mission condition (1 success, 0 failure)", self.shared_dict['mission_condition_writer'].get(), current_eps)
                        # write energy cunsumption for all drone (added them together)
                        writer.add_scalar("Energy Consumption", self.shared_dict['total_energy_consump_writer'].get(), current_eps)
                        # write each component of attacker's reward
                        writer.add_scalar("Attacker's Reward (component 0)",
                                          self.shared_dict['att_reward_0_writer'].get(), current_eps)
                        writer.add_scalar("Attacker's Reward (component 1)",
                                          self.shared_dict['att_reward_1_writer'].get(), current_eps)
                        writer.add_scalar("Attacker's Reward (component 2)",
                                          self.shared_dict['att_reward_2_writer'].get(), current_eps)
                        writer.add_scalar("Defender's Reward (component 0)",
                                          self.shared_dict['def_reward_0_writer'].get(), current_eps)
                        writer.add_scalar("Defender's Reward (component 1)",
                                          self.shared_dict['def_reward_1_writer'].get(), current_eps)
                        writer.add_scalar("Defender's Reward (component 2)",
                                          self.shared_dict['def_reward_2_writer'].get(), current_eps)
                        writer.add_scalar("Max Mission Duration", self.env.system.mission_duration_max, current_eps)

                    ep_counter += 1

            if not self.exist_model:  # no need to decay lr and epsilon for a trained model
                self.scheduler.step()  # update learning rate each episode (each agent will update lr independently)
                self.lnet.epsilon_decay_step()  # update epsilon each episode
            print(self.name, 'episode ', self.g_ep.value, 'defender reward %.1f' % score_def, 'attacker reward %.1f' % score_att)
        # self.res_queue.put(None)

        if self.is_custom_env: self.env.close_env()

        if writer is not None:
            writer.flush()
            writer.close()  # close SummaryWriter of TensorBoard
        return