'''
Project     ：gym-drones 
File        ：A3C_model_4.py.py
Author      ：Zelin Wan
Date        ：9/20/22
Description : The agent that will be used by 'A3C_train_agent_optuna_4.py'
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
import threading

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

        self.eval()
        # logits, _ = self.forward(s)
        # print("obs", obs, "self.pi_net", self.pi_net)
        logits = self.pi_net(obs)
        prob = F.softmax(logits, dim=1).data
        # print("Pi Net Prob:", prob)
        m = self.distribution(prob)
        return m.sample().numpy()[0]

    def loss_func(self, s, a, v_t):
        # print("v_t", v_t)
        self.train()
        # logits, values = self.forward(s)
        logits = self.pi_net(s)
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

class MultiAgent(mp.Process):
    def __init__(self, gnet_def, gnet_att, opt_def, opt_att, shared_dict, lr_decay, gamma, epsilon, epsilon_decay, MAX_EP, fixed_seed, trial, name_id: int, player='def', exist_model=False, is_custom_env=True, miss_dur=30, target_size=5):
        super(MultiAgent, self).__init__()
        self.name_id = name_id
        self.name = 'w%02i' % name_id
        self.gnet_def = gnet_def
        self.gnet_att = gnet_att
        self.opt_def = opt_def
        self.opt_att = opt_att
        self.shared_dict = shared_dict
        self.g_ep = shared_dict['global_ep']
        self.g_r_list = shared_dict['glob_r_list']
        self.lr_decay = lr_decay
        self.exist_model = exist_model
        self.is_custom_env = is_custom_env
        self.miss_dur = miss_dur
        self.target_size = target_size
        if is_custom_env:
            self.env = HyperGameSim(fixed_seed=fixed_seed, miss_dur=self.miss_dur, target_size=self.target_size)
            # self.N_S = self.env.observation_space[player].shape[0]
            # self.N_A = self.env.action_space[player].n
            self.input_dims_def = self.env.observation_space['def'].shape[0]
            self.n_actions_def = self.env.action_space['def'].n
            self.input_dims_att = self.env.observation_space['att'].shape[0]
            self.n_actions_att = self.env.action_space['att'].n
        else:
            self.env = gym.make('CartPole-v1')
            self.input_dims_def = self.env.observation_space.shape[0]
            # self.input_dims_def = self.env.observation_space.n
            self.n_actions_def = self.env.action_space.n
        self.lnet_def = ActorCritic(input_dims=self.input_dims_def, n_actions=self.n_actions_def, epsilon=epsilon, epsilon_decay=epsilon_decay,
                                pi_net_struc=gnet_def.pi_net_struc,
                                v_net_struct=gnet_def.v_net_struct, fixed_seed=fixed_seed)  # local network for defender
        # load pre-trained model's parameters
        if self.exist_model:
            path = "trained_model/def/trained_A3C"
            self.lnet_def.load_state_dict(torch.load(path))
            self.lnet_def.eval()

        if is_custom_env:
            self.lnet_att = ActorCritic(input_dims=self.input_dims_att, n_actions=self.n_actions_att, epsilon=epsilon,
                                        epsilon_decay=epsilon_decay,
                                        pi_net_struc=gnet_att.pi_net_struc,
                                        v_net_struct=gnet_att.v_net_struct, fixed_seed=fixed_seed)  # local network for attacker
            # load pre-trained model's parameters
            if self.exist_model:
                path = "trained_model/att/trained_A3C"
                self.lnet_att.load_state_dict(torch.load(path))
                self.lnet_att.eval()

        self.gamma = gamma
        self.MAX_EP = MAX_EP
        self.trial = trial
        self.player = player
        self.scheduler_def = LambdaLR(self.opt_def, lr_lambda=self.lambda_function)
        if is_custom_env:
            self.scheduler_att = LambdaLR(self.opt_att, lr_lambda=self.lambda_function)

    def lambda_function(self, epoch):  # epoch increase one when scheduler.step() is called
        return self.lr_decay ** epoch

    def run(self) -> None:
        # ======== Create Writer for TensorBoard ========
        # run 'tensorboard --logdir=runs' in terminal to start TensorBoard.
        if self.name_id == 0:
            if self.trial is not None:
                trial_num_str = str(self.trial.number)
            else:
                trial_num_str = "None"

            if self.shared_dict["on_server"]:
                path = "/home/zelin/Drone/data/"+str(self.miss_dur)+"_"+str(self.target_size)+"/"
            else:
                path = "data/"+str(self.miss_dur)+"_"+str(self.target_size)+"/"
            writer = SummaryWriter(log_dir=path + "runs_DefAtt/each_run_" + self.shared_dict["start_time"] + "-DefAtt-" + "-Trial_" + trial_num_str + "-eps")
            # writer = None
            # print("creating writer", "runs_"+self.player+"/each_run_" + self.shared_dict["start_time"] + "-" + self.player + "-" + "-Trial_" + trial_num_str + "-eps")

        else:
            writer = None

        ep_counter = 0
        while self.g_ep.value < self.MAX_EP:
            # print("Number of active threads:", threading.active_count())
            with self.g_ep.get_lock():
                temp_ep = self.g_ep.value

            if self.is_custom_env:
                obs_out = self.env.reset(miss_dur=self.miss_dur, target_size=self.target_size)
                # if self.player == "att":
                #     obs = obs_out['att']
                # elif self.player == "def":
                #     obs = obs_out['def']
                # else:
                #     raise Exception("invalide 'player_name'")
                obs_def = obs_out['def']
                obs_att = obs_out['att']
            else:
                obs_def = self.env.reset()


            buffer_s_def, buffer_a_def, buffer_r_def = [], [], []
            buffer_s_att, buffer_a_att, buffer_r_att = [], [], []
            total_loss_set_def = []
            c_loss_set_def = []
            a_loss_set_def = []
            total_loss_set_att = []
            c_loss_set_att = []
            a_loss_set_att = []
            att_succ_rate_set = []
            MDHD_active_set = []
            MDHD_connect_RLD_set = []
            mission_complete_rate_set = []
            remaining_time_set = []
            energy_HD_set = []
            energy_MD_set = []
            att_reward_0 = 0
            att_reward_1 = 0
            att_reward_2 = 0
            def_reward_0 = 0
            def_reward_1 = 0
            def_reward_2 = 0
            def_stra_counter = np.zeros(10)
            att_stra_counter = np.zeros(10)
            score_def = 0
            score_att = 0
            total_step = 1
            done = False
            while not done:
                # if self.name_id == 0:
                #     self.env.render()

                # choose action
                # action = self.lnet.choose_action(v_wrap(obs[None, :]))
                # assign action to attacker or defender
                # print("Defender")
                action_def = self.lnet_def.choose_action(v_wrap(obs_def[None, :]))
                # print("action_def", action_def)
                if self.is_custom_env:
                    # print("Attacker")
                    action_att = self.lnet_att.choose_action(v_wrap(obs_att[None, :]))
                    # print("action_att", action_att)

                if self.is_custom_env:  # for Drone environment

                    # if self.player == "att":
                    #     action_att = action
                    # elif self.player == "def":
                    #     action_def = action
                    # else:
                    #     print("Error: player is not specified, using action for defender")
                    #     action_def = action
                    # interaction with environment
                    the_obs_, the_reward, done, info = self.env.step(action_def=action_def, action_att=action_att)
                    # the_obs_, the_reward, done, info = self.env.step()      # TODO: this is a test
                    # extract different reward and observation for attacker and defender
                    # if self.player == "att":
                    #     reward = the_reward['att']
                    #     oppo_reward = the_reward['def']
                    #     obs_ = the_obs_['att']
                    # elif self.player == "def":
                    #     reward = the_reward['def']
                    #     oppo_reward = the_reward['att']
                    #     obs_ = the_obs_['def']
                    # else:
                    #     raise Exception("invalide 'player_name'")
                    reward_def = the_reward['def']
                    reward_att = the_reward['att']
                    obs_new_def = the_obs_['def']
                    obs_new_att = the_obs_['att']
                    # add data from each step (for custom env only)
                    # att_succ_rate_set.append(info['att_succ_rate'])
                    if info['att_counter']:  # this avoids dividing by zero
                        att_succ_rate_set.append(info['att_succ_counter'] / info['att_counter'])
                    att_reward_0 += info["att_reward_0"]
                    att_reward_1 += info["att_reward_1"]
                    att_reward_2 += info["att_reward_2"]
                    def_reward_0 += info["def_reward_0"]
                    def_reward_1 += info["def_reward_1"]
                    def_reward_2 += info["def_reward_2"]
                    def_stra_counter[info["action_def"]] += 1
                    att_stra_counter[info["action_att"]] += 1
                    MDHD_active_set.append(info["MDHD_active_num"])
                    MDHD_connect_RLD_set.append(info["MDHD_connected_num"])
                    mission_complete_rate_set.append(info["mission_complete_rate"])
                    remaining_time_set.append(info["remaining_time"])
                    energy_HD_set.append(info["energy_HD"])
                    energy_MD_set.append(info["energy_MD"])
                else:
                    # since gym standard environment only need one action, we use network for defender to control
                    obs_new_def, reward_def, done, _ = self.env.step(action_def)
                    oppo_reward = 0  # no opponent, assign 0 for avoiding error

                # if done: reward = -1
                # add data from each step
                # score += reward
                # oppo_score += oppo_reward
                score_def += reward_def
                if self.is_custom_env:
                    score_att += reward_att

                if not self.exist_model:  # no need to train a trained model
                    # add to memory
                    buffer_a_def.append(action_def)
                    buffer_s_def.append(obs_def)
                    buffer_r_def.append(reward_def)
                    if self.is_custom_env:
                        buffer_a_att.append(action_att)
                        buffer_s_att.append(obs_att)
                        buffer_r_att.append(reward_att)

                    # update global and assign to local net
                    if total_step % 5 == 0 or done:
                        # sync
                        total_loss_def, c_loss_def, a_loss_def = push_and_pull(self.opt_def, self.lnet_def, self.gnet_def,
                                                                               done, obs_new_def, buffer_s_def,
                                                                               buffer_a_def, buffer_r_def, self.gamma)
                        if self.is_custom_env:
                            total_loss_att, c_loss_att, a_loss_att = push_and_pull(self.opt_def, self.lnet_def,
                                                                                   self.gnet_def, done, obs_new_def,
                                                                                   buffer_s_def, buffer_a_def, buffer_r_def,
                                                                                   self.gamma)
                        # save loss data for display
                        total_loss_set_def.append(total_loss_def.item())
                        c_loss_set_def.append(c_loss_def.item())
                        a_loss_set_def.append(a_loss_def.item())
                        if self.is_custom_env:
                            total_loss_set_att.append(total_loss_att.item())
                            c_loss_set_att.append(c_loss_att.item())
                            a_loss_set_att.append(a_loss_att.item())

                        # clear memory
                        buffer_s_def, buffer_a_def, buffer_r_def = [], [], []
                        if self.is_custom_env:
                            buffer_s_att, buffer_a_att, buffer_r_att = [], [], []




                # obs = obs_
                obs_def = obs_new_def
                if self.is_custom_env:
                    obs_att = obs_new_att

                total_step += 1

            score_def_avg = score_def / total_step
            score_att_avg = score_att / total_step
            # save data to shared dictionary for tensorboard
            with self.g_ep.get_lock():
                self.g_ep.value += 1
                self.shared_dict['eps_writer'].put(temp_ep)
                self.g_r_list.append(score_def) # use defender's reward to guild optuna
                self.shared_dict["lr_writer"].put(self.opt_def.param_groups[0]['lr'])
                self.shared_dict["epsilon_writer"].put(self.lnet_def.epsilon)
                self.shared_dict["score_def_writer"].put(score_def)
                self.shared_dict["score_def_avg_writer"].put(score_def_avg)
                self.shared_dict['t_loss_def_writer'].put(sum(total_loss_set_def) / len(total_loss_set_def) if len(total_loss_set_def) else 0)
                self.shared_dict['c_loss_def_writer'].put(sum(c_loss_set_def) / len(c_loss_set_def) if len(c_loss_set_def) else 0)
                self.shared_dict['a_loss_def_writer'].put(sum(a_loss_set_def) / len(a_loss_set_def) if len(a_loss_set_def) else 0)
                self.shared_dict['def_stra_count_writer'].put(def_stra_counter)
                self.shared_dict['att_stra_count_writer'].put(att_stra_counter)
                if self.is_custom_env:
                    self.shared_dict["score_att_writer"].put(score_att)
                    self.shared_dict["score_att_avg_writer"].put(score_att_avg)
                    self.shared_dict['t_loss_att_writer'].put(sum(total_loss_set_att) / len(total_loss_set_att) if len(total_loss_set_att) else 0)
                    self.shared_dict['c_loss_att_writer'].put(sum(c_loss_set_att) / len(c_loss_set_att) if len(c_loss_set_att) else 0)
                    self.shared_dict['a_loss_att_writer'].put(sum(a_loss_set_att) / len(a_loss_set_att) if len(a_loss_set_att) else 0)
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
                    self.shared_dict['MDHD_active_num_writer'].put(sum(MDHD_active_set) / len(MDHD_active_set) if len(MDHD_active_set) else 0)
                    self.shared_dict['MDHD_connect_RLD_num_writer'].put(sum(MDHD_connect_RLD_set) / len(MDHD_connect_RLD_set) if len(MDHD_connect_RLD_set) else 0)
                    self.shared_dict['mission_complete_rate_writer'].put(sum(mission_complete_rate_set) / len(mission_complete_rate_set) if len(mission_complete_rate_set) else 0)
                    self.shared_dict['remaining_time_writer'].put(sum(remaining_time_set) / len(remaining_time_set) if len(remaining_time_set) else 0)
                    self.shared_dict['energy_HD_writer'].put(sum(energy_HD_set) / len(energy_HD_set) if len(energy_HD_set) else 0)
                    self.shared_dict['energy_MD_writer'].put(sum(energy_MD_set) / len(energy_MD_set) if len(energy_MD_set) else 0)

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
                    writer.add_scalar("Defender's Total Loss", self.shared_dict['t_loss_def_writer'].get(), current_eps)
                    writer.add_scalar("Defender's Critic Loss", self.shared_dict['c_loss_def_writer'].get(), current_eps)
                    writer.add_scalar("Defender's Actor Loss", self.shared_dict['a_loss_def_writer'].get(), current_eps)
                    writer.add_scalar("Attacker's Total Loss", self.shared_dict['t_loss_att_writer'].get(), current_eps)
                    writer.add_scalar("Attacker's Critic Loss", self.shared_dict['c_loss_att_writer'].get(), current_eps)
                    writer.add_scalar("Attacker's Actor Loss", self.shared_dict['a_loss_att_writer'].get(), current_eps)
                    # write strategy counter
                    def_stra_counter = self.shared_dict['def_stra_count_writer'].get()
                    count_sum = np.sum(def_stra_counter)
                    def_stra_counter = def_stra_counter / count_sum
                    for i in range(10):
                        writer.add_scalar("Defender Strategy (" + str(i) + ") Freq.", def_stra_counter[i], current_eps)
                    att_stra_counter = self.shared_dict['att_stra_count_writer'].get()
                    count_sum = np.sum(att_stra_counter)
                    att_stra_counter = att_stra_counter / count_sum
                    for i in range(10):
                        writer.add_scalar("Attacker Strategy (" + str(i) + ") Freq.", att_stra_counter[i], current_eps)
                    # write mission completion rate
                    if self.is_custom_env:
                        # write attack success rate
                        writer.add_scalar("Attack Success Rate", self.shared_dict['att_succ_rate_writer'].get(),
                                          current_eps)
                        # write Ratio of Mission Completion
                        writer.add_scalar("Ratio of Mission Completion",
                                          self.shared_dict['scan_percent_writer'].get(), current_eps)
                        # # write missin result
                        # writer.add_scalar("Mission condition (1 success, 0 failure)", self.shared_dict['mission_condition_writer'].get(), current_eps)
                        # write energy cunsumption for all drone (added them together)
                        writer.add_scalar("Energy Consumption", self.shared_dict['total_energy_consump_writer'].get(),
                                          current_eps)
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
                        # write other metrics
                        writer.add_scalar("Max Mission Duration", self.env.system.mission_duration_max, current_eps)
                        writer.add_scalar("Number of Cell to Scan", self.env.system.map_cell_number * self.env.system.map_cell_number, current_eps)
                        writer.add_scalar("Average Active MD+HD Number", self.shared_dict['MDHD_active_num_writer'].get(), current_eps)
                        writer.add_scalar("Average Connect_RLD MD+HD Number", self.shared_dict['MDHD_connect_RLD_num_writer'].get(), current_eps)
                        writer.add_scalar("Average Mission Complete Rate", self.shared_dict['mission_complete_rate_writer'].get(), current_eps)
                        writer.add_scalar("Average Remaining Time Writer", self.shared_dict['remaining_time_writer'].get(), current_eps)
                        writer.add_scalar("Average Energy HD", self.shared_dict['energy_HD_writer'].get(), current_eps)
                        writer.add_scalar("Average Energy MD", self.shared_dict['energy_MD_writer'].get(), current_eps)

                    ep_counter += 1

            if not self.exist_model:  # no need to decay lr and epsilon for a trained model
                self.scheduler_def.step()  # update learning rate each episode (each agent will update lr independently)
                self.lnet_def.epsilon_decay_step()  # update epsilon each episode

            if self.is_custom_env:
                self.scheduler_att.step()  # update learning rate each episode (each agent will update lr independently)
                self.lnet_att.epsilon_decay_step()  # update epsilon each episode

            print(self.name, 'episode ', self.g_ep.value, 'defender reward %.1f' % score_def, 'attacker reward %.1f' % score_att)
        # self.res_queue.put(None)

        if self.is_custom_env: self.env.close_env()

        if writer is not None:
            writer.flush()
            writer.close()  # close SummaryWriter of TensorBoard
        return
