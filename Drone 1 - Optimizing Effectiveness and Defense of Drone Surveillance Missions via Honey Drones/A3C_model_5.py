'''
Project     ：gym-drones 
File        ：A3C_model_5.py
Author      ：Zelin Wan
Date        ：12/2/22
Description : 
'''

import threading
import torch
import torch.nn as nn
from utils import v_wrap, set_init, push_and_pull, record
import torch.nn.functional as F
import torch.multiprocessing as mp
import gym
import time
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from Gym_HoneyDrone_Defender_and_Attacker import HyperGameSim


class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions, epsilon=0.1, epsilon_decay=0.9, pi_net_struc=None, v_net_struct=None,
                 fixed_seed=True):
        super(ActorCritic, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.epsilon = epsilon
        print("self.epsilon", self.epsilon)
        self.epsilon_decay = epsilon_decay
        if pi_net_struc is None:
            pi_net_struc = [128]  # default 64, 64 hidden layers if not specify
        else:
            self.pi_net_struc = pi_net_struc
        if v_net_struct is None:
            v_net_struct = [128]  # default 64, 64 hidden layers if not specify
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
        # print("prob", torch.sum(torch.isnan(prob)))
        # if torch.sum(torch.isnan(prob)):    # avoid 'nan' error. When [nan], randomly choose an action
        #     return torch.randint(0, len(prob), (1,)).numpy()[0]
        # else:
        #     m = self.distribution(prob)
        #     return m.sample().numpy()[0]
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


class Agent(mp.Process):
    def __init__(self, gnet_def, gnet_att, opt_def, opt_att, shared_dict, config_def, config_att, fixed_seed, trial,
                 name_id: int, exp_scheme=0, player='def', exist_model=False, is_custom_env=True, defense_strategy=0, miss_dur=30,
                 target_size=5, max_att_budget=5, num_HD=2):
        super(Agent, self).__init__()
        self.name_id = name_id
        self.name = 'w%02i' % name_id
        self.exp_scheme = exp_scheme
        self.gnet_def = gnet_def
        self.gnet_att = gnet_att
        self.opt_def = opt_def
        self.opt_att = opt_att
        self.shared_dict = shared_dict
        self.g_ep = shared_dict['global_ep']
        self.g_r_list = shared_dict['glob_r_list']
        self.lr_decay_def = config_def["LR_decay"]
        self.lr_decay_att = config_att["LR_decay"]
        self.exist_model = exist_model
        self.is_custom_env = is_custom_env
        self.defense_strategy = defense_strategy
        self.miss_dur = miss_dur
        self.target_size = target_size
        self.max_att_budget = max_att_budget

        self.num_HD = num_HD
        if is_custom_env:
            self.env = HyperGameSim(fixed_seed=fixed_seed, miss_dur=miss_dur, target_size=target_size,
                                    max_att_budget=max_att_budget, num_HD=num_HD, defense_strategy=self.defense_strategy)
            self.input_dims_def = self.env.observation_space['def'].shape[0]
            self.n_actions_def = self.env.action_space['def'].n
            self.input_dims_att = self.env.observation_space['att'].shape[0]
            self.n_actions_att = self.env.action_space['att'].n
            # update setting for sensitivity analysis
        else:
            self.env = gym.make('CartPole-v1')
            self.input_dims_def = self.env.observation_space.shape[0]
            # self.N_S = self.env.observation_space.n
            self.n_actions_def = self.env.action_space.n

        # defender's local AC
        self.lnet_def = ActorCritic(input_dims=self.input_dims_def, n_actions=self.n_actions_def,
                                    epsilon=config_def["epsilon"],
                                    epsilon_decay=config_def["epsilon_decay"], pi_net_struc=gnet_def.pi_net_struc,
                                    v_net_struct=gnet_def.v_net_struct,
                                    fixed_seed=fixed_seed)  # local network for defender
        if self.exist_model:
            # load pre-trained model's parameters
            path = "trained_model/DefAtt/def/trained_A3C"
            self.lnet_def.load_state_dict(torch.load(path))
            self.lnet_def.eval()

        # attacker's local AC
        self.lnet_att = ActorCritic(input_dims=self.input_dims_att, n_actions=self.n_actions_att,
                                    epsilon=config_att["epsilon"],
                                    epsilon_decay=config_att["epsilon_decay"], pi_net_struc=gnet_att.pi_net_struc,
                                    v_net_struct=gnet_att.v_net_struct,
                                    fixed_seed=fixed_seed)  # local network for attacker
        if self.exist_model:
            # load pre-trained model's parameters
            path = "trained_model/DefAtt/att/trained_A3C"
            self.lnet_att.load_state_dict(torch.load(path))
            self.lnet_att.eval()

        self.gamma_def = config_def["gamma"]
        self.gamma_att = config_att["gamma"]
        self.MAX_EP = config_def["glob_episode_thred"]  # defender and attacker should have the same
        # 'glob_episode_thred', so I didn't make them different here
        self.trial = trial
        self.player = player
        self.scheduler_def = LambdaLR(self.opt_def, lr_lambda=self.lambda_function_def)
        self.scheduler_att = LambdaLR(self.opt_att, lr_lambda=self.lambda_function_att)

    def lambda_function_def(self, epoch):  # epoch increase one when scheduler_def.step() is called
        return self.lr_decay_def ** epoch

    def lambda_function_att(self, epoch):  # epoch increase one when scheduler_att.step() is called
        return self.lr_decay_att ** epoch

    def run(self):
        # get start time of this training (for all episodes)
        train_start_time = time.time()
        # ======== Create Writer for TensorBoard ========
        # run 'tensorboard --logdir=runs' in terminal to start TensorBoard.
        if self.name_id == 0:
            # make the first thread as the main writer
            if self.trial is not None:
                trial_num_str = str(self.trial.number)
            else:
                trial_num_str = "None"

            if self.shared_dict["on_server"]:
                # set path for server
                path = "/home/zelin/Drone/data/" + str(self.miss_dur) + "_" + str(self.max_att_budget) + "_" + str(
                    self.num_HD) + "_" + str(self.defense_strategy) + "/"
            else:
                # set path for my laptop
                path = "data/" + str(self.miss_dur) + "_" + str(self.max_att_budget) + "_" + str(self.num_HD) + "_" + str(self.defense_strategy) + "/"
            writer = SummaryWriter(
                log_dir=path + "runs_" + self.player + "/each_run_" + self.shared_dict["start_time"] + "-" +
                        self.player + "-" + "-Trial_" + trial_num_str + "-eps")
        else:
            writer = None

        ep_counter = 0

        while self.g_ep.value < self.MAX_EP:
            # run until episode limit reached
            # print("Number of active threads:", threading.active_count())
            with self.g_ep.get_lock():
                temp_ep = self.g_ep.value

            if self.is_custom_env:
                obs_out = self.env.reset(miss_dur=self.miss_dur, target_size=self.target_size,
                                         max_att_budget=self.max_att_budget, num_HD=self.num_HD)
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
            att_succ_counter_set = []
            att_counter_set = []
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
            score_att = 0
            score_def = 0
            total_step = 1
            done = False
            # get the start time of this episode
            start_time = time.time()

            while not done:
                # run until environment end

                # choose action
                action_def = None
                action_att = None
                if self.exp_scheme == 0:  # A-random D-a3c
                    action_def = self.lnet_def.choose_action(v_wrap(obs_def[None, :]))
                elif self.exp_scheme == 1:  # A-a3c D-random
                    action_att = self.lnet_att.choose_action(v_wrap(obs_att[None, :]))
                elif self.exp_scheme == 2:  # A-a3c D-a3c
                    action_def = self.lnet_def.choose_action(v_wrap(obs_def[None, :]))
                    action_att = self.lnet_att.choose_action(v_wrap(obs_att[None, :]))
                else:  # A-random D-random
                    pass

                # if use IDS, then the defender will choose fixed signal strength
                if self.defense_strategy == 1 or self.defense_strategy == 2 or self.defense_strategy == 3:
                    action_def = torch.tensor(3, dtype=torch.int64)

                if self.is_custom_env:  # for Drone environment
                    # interaction with environment
                    the_obs_, the_reward, done, info = self.env.step(action_def=action_def, action_att=action_att)

                    reward_def = the_reward['def']
                    reward_att = the_reward['att']
                    obs_new_def = the_obs_['def']
                    obs_new_att = the_obs_['att']
                    action_def = info["action_def"]
                    action_att = info["action_att"]

                    # if self.player == "att":
                    #     obs_ = the_obs_['att']
                    # elif self.player == "def":
                    #     obs_ = the_obs_['def']
                    # else:
                    #     raise Exception("invalide 'player_name'")

                    # add data from each step (for custom env only)
                    # att_succ_rate_set.append(info['att_succ_rate'])
                    att_succ_counter_set.append(info['att_succ_counter'])
                    att_counter_set.append(info['att_counter'])
                    if info['att_counter']:  # this avoids dividing by zero
                        att_succ_rate_set.append(info['att_succ_counter'] / info['att_counter'])
                    att_reward_0 += info["att_reward_0"]
                    att_reward_1 += info["att_reward_1"]
                    att_reward_2 += info["att_reward_2"]
                    def_reward_0 += info["def_reward_0"]
                    def_reward_1 += info["def_reward_1"]
                    def_reward_2 += info["def_reward_2"]
                    def_stra_counter[action_def] += 1
                    att_stra_counter[action_att] += 1
                    MDHD_active_set.append(info["MDHD_active_num"])
                    MDHD_connect_RLD_set.append(info["MDHD_connected_num"])
                    mission_complete_rate_set.append(info["mission_complete_rate"])
                    remaining_time_set.append(info["remaining_time"])
                    energy_HD_set.append(info["energy_HD"])
                    energy_MD_set.append(info["energy_MD"])
                else:
                    obs_new_def, reward_def, done, _ = self.env.step(
                        action_def)  # assign reward to reward_def, ignore reward_att
                    reward_att = 0
                score_def += reward_def
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
                        total_loss_def, c_loss_def, a_loss_def = push_and_pull(self.opt_def, self.lnet_def,
                                                                               self.gnet_def, done, obs_new_def,
                                                                               buffer_s_def,
                                                                               buffer_a_def, buffer_r_def,
                                                                               self.gamma_def)
                        if self.is_custom_env:
                            total_loss_att, c_loss_att, a_loss_att = push_and_pull(self.opt_att, self.lnet_att,
                                                                                   self.gnet_att, done, obs_new_att,
                                                                                   buffer_s_att,
                                                                                   buffer_a_att, buffer_r_att,
                                                                                   self.gamma_att)

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

                obs_def = obs_new_def
                if self.is_custom_env:
                    obs_att = obs_new_att
                total_step += 1
            # game done

            score_def_avg = score_def / total_step
            score_att_avg = score_att / total_step

            # get the end time
            end_time = time.time()
            # calculate the total time taken in seconds
            total_time = end_time - start_time
            # calculate the total time taken for training
            train_end_time = time.time()
            total_time_taken = train_end_time - train_start_time

            # save data to shared dictionary for tensorboard
            with self.g_ep.get_lock():
                self.g_ep.value += 1
                self.shared_dict['eps_writer'].put(temp_ep)
                if self.exp_scheme == 0 or self.exp_scheme == 2:  # 'A-random D-a3c' and 'A-a3c D-a3c' use defender's reward to guid optuna
                    self.g_r_list.append(score_def)
                elif self.exp_scheme == 1:  # 'A-a3c D-random' uses attacker's reward to guid optuna
                    self.g_r_list.append(score_att)
                else:  # 'A-random D-random' does need optuna
                    pass
                self.shared_dict["lr_def_writer"].put(self.opt_def.param_groups[0]['lr'])
                self.shared_dict["lr_att_writer"].put(self.opt_att.param_groups[0]['lr'])
                self.shared_dict["epsilon_def_writer"].put(self.lnet_def.epsilon)
                self.shared_dict["epsilon_att_writer"].put(self.lnet_att.epsilon)
                self.shared_dict["score_def_writer"].put(score_def)
                self.shared_dict["score_att_writer"].put(score_att)
                self.shared_dict["score_def_avg_writer"].put(score_def_avg)
                self.shared_dict["score_att_avg_writer"].put(score_att_avg)
                self.shared_dict['t_loss_def_writer'].put(
                    sum(total_loss_set_def) / len(total_loss_set_def) if len(total_loss_set_def) else 0)
                self.shared_dict['c_loss_def_writer'].put(
                    sum(c_loss_set_def) / len(c_loss_set_def) if len(c_loss_set_def) else 0)
                self.shared_dict['a_loss_def_writer'].put(
                    sum(a_loss_set_def) / len(a_loss_set_def) if len(a_loss_set_def) else 0)
                self.shared_dict['t_loss_att_writer'].put(
                    sum(total_loss_set_att) / len(total_loss_set_att) if len(total_loss_set_att) else 0)
                self.shared_dict['c_loss_att_writer'].put(
                    sum(c_loss_set_att) / len(c_loss_set_att) if len(c_loss_set_att) else 0)
                self.shared_dict['a_loss_att_writer'].put(
                    sum(a_loss_set_att) / len(a_loss_set_att) if len(a_loss_set_att) else 0)
                self.shared_dict['def_stra_count_writer'].put(def_stra_counter)
                self.shared_dict['att_stra_count_writer'].put(att_stra_counter)
                # put running time to shared dictionary
                self.shared_dict['running_time_writer'].put(total_time)
                self.shared_dict['running_time_taken_writer'].put(total_time_taken)
                if self.is_custom_env:
                    self.shared_dict['att_succ_counter_writer'].put(
                        sum(att_succ_counter_set))  # the total number of success attack in one episode
                    self.shared_dict['att_succ_counter_ave_writer'].put(
                        sum(att_succ_counter_set) / len(att_succ_counter_set) if len(
                            att_succ_counter_set) else 0)  # the average number of success attack in one step
                    self.shared_dict['att_counter_writer'].put(
                        sum(att_counter_set))  # the total number of attack behavior in one episode
                    self.shared_dict['att_counter_ave_writer'].put(sum(att_counter_set) / len(att_counter_set) if len(
                        att_counter_set) else 0)  # the average number of attack behavior in one step
                    self.shared_dict['att_succ_rate_writer'].put(
                        sum(att_succ_rate_set) / len(att_succ_rate_set) if len(att_succ_rate_set) else 0)
                    self.shared_dict['mission_condition_writer'].put(info["mission_condition"])
                    self.shared_dict['total_energy_consump_writer'].put(info["total_energy_consump"])
                    self.shared_dict['scan_percent_writer'].put(info["scan_percent"])
                    self.shared_dict['att_reward_0_writer'].put(att_reward_0)
                    self.shared_dict['att_reward_1_writer'].put(att_reward_1)
                    self.shared_dict['att_reward_2_writer'].put(att_reward_2)
                    self.shared_dict['def_reward_0_writer'].put(def_reward_0)
                    self.shared_dict['def_reward_1_writer'].put(def_reward_1)
                    self.shared_dict['def_reward_2_writer'].put(def_reward_2)
                    self.shared_dict['MDHD_active_num_writer'].put(
                        sum(MDHD_active_set) / len(MDHD_active_set) if len(MDHD_active_set) else 0)
                    self.shared_dict['MDHD_connect_RLD_num_writer'].put(
                        sum(MDHD_connect_RLD_set) / len(MDHD_connect_RLD_set) if len(MDHD_connect_RLD_set) else 0)
                    self.shared_dict['mission_complete_rate_writer'].put(
                        sum(mission_complete_rate_set) / len(mission_complete_rate_set) if len(
                            mission_complete_rate_set) else 0)
                    self.shared_dict['remaining_time_writer'].put(
                        sum(remaining_time_set) / len(remaining_time_set) if len(remaining_time_set) else 0)
                    self.shared_dict['energy_HD_writer'].put(
                        sum(energy_HD_set) / len(energy_HD_set) if len(energy_HD_set) else 0)
                    self.shared_dict['energy_MD_writer'].put(
                        sum(energy_MD_set) / len(energy_MD_set) if len(energy_MD_set) else 0)
                    # put recorded_max_RLD_down_time to shared dictionary
                    self.shared_dict['recorded_max_RLD_down_time_writer'].put(info["recorded_max_RLD_down_time"])
                    # put number of alive MD to shared dictionary
                    self.shared_dict['alive_MD_num_writer'].put(info["alive_MD_num"])
                    # put number of alive HD to shared dictionary
                    self.shared_dict['alive_HD_num_writer'].put(info["alive_HD_num"])


            # ==== tensorboard writer ====
            # Only agent (index 0) can write to tensorboard
            if writer is not None:
                while not self.shared_dict['eps_writer'].empty():
                    # use episode as index of tensorboard
                    current_eps = self.shared_dict['eps_writer'].get()
                    current_eps = ep_counter
                    # write score
                    writer.add_scalar("Accumulated Reward Defender", self.shared_dict["score_def_writer"].get(),
                                      current_eps)
                    writer.add_scalar("Accumulated Reward Attacker", self.shared_dict["score_att_writer"].get(),
                                      current_eps)
                    writer.add_scalar("Averaged Reward Defender", self.shared_dict["score_def_avg_writer"].get(),
                                      current_eps)
                    writer.add_scalar("Averaged Reward Attacker", self.shared_dict["score_att_avg_writer"].get(),
                                      current_eps)
                    # write lr
                    writer.add_scalar("Learning rate (def)", self.shared_dict["lr_def_writer"].get(), current_eps)
                    writer.add_scalar("Learning rate (att)", self.shared_dict["lr_att_writer"].get(), current_eps)
                    # write epsilon
                    writer.add_scalar("Epsilon (random action probability) (def)",
                                      self.shared_dict["epsilon_def_writer"].get(),
                                      current_eps)
                    writer.add_scalar("Epsilon (random action probability) (att)",
                                      self.shared_dict["epsilon_att_writer"].get(),
                                      current_eps)
                    # write mission time (step)
                    writer.add_scalar("Mission Time (step)", total_step, current_eps)
                    # write loss
                    writer.add_scalar("Defender's Total Loss", self.shared_dict['t_loss_def_writer'].get(), current_eps)
                    writer.add_scalar("Defender's Critic Loss", self.shared_dict['c_loss_def_writer'].get(),
                                      current_eps)
                    writer.add_scalar("Defender's Actor Loss", self.shared_dict['a_loss_def_writer'].get(), current_eps)
                    writer.add_scalar("Attacker's Total Loss", self.shared_dict['t_loss_att_writer'].get(), current_eps)
                    writer.add_scalar("Attacker's Critic Loss", self.shared_dict['c_loss_att_writer'].get(),
                                      current_eps)
                    writer.add_scalar("Attacker's Actor Loss", self.shared_dict['a_loss_att_writer'].get(), current_eps)
                    # write strategy counter
                    # for defender
                    def_stra_counter = self.shared_dict['def_stra_count_writer'].get()
                    count_sum = np.sum(def_stra_counter)
                    def_stra_counter = def_stra_counter / count_sum
                    for i in range(10):
                        writer.add_scalar("Defender Strategy (" + str(i) + ") Freq.", def_stra_counter[i], current_eps)
                    # for attacker
                    att_stra_counter = self.shared_dict['att_stra_count_writer'].get()
                    count_sum = np.sum(att_stra_counter)
                    att_stra_counter = att_stra_counter / count_sum
                    for i in range(10):
                        writer.add_scalar("Attacker Strategy (" + str(i) + ") Freq.", att_stra_counter[i], current_eps)
                    # write mission completion rate
                    # write running time to tensorboard
                    writer.add_scalar("Running Time (s)", self.shared_dict['running_time_writer'].get(), current_eps)
                    # write running time taken to tensorboard
                    writer.add_scalar("Total Running Time Taken (s)", self.shared_dict['running_time_taken_writer'].get(), current_eps)

                    if self.is_custom_env:
                        # write attack related data
                        writer.add_scalar("Attack Success Counter", self.shared_dict['att_succ_counter_writer'].get(), current_eps)
                        writer.add_scalar("Attack Success Counter (step averaged)", self.shared_dict['att_succ_counter_ave_writer'].get(),
                                          current_eps)
                        writer.add_scalar("Attack Launched Counter", self.shared_dict['att_counter_writer'].get(),
                                          current_eps)
                        writer.add_scalar("Attack Launched Counter (step averaged)", self.shared_dict['att_counter_ave_writer'].get(),
                                          current_eps)
                        writer.add_scalar("Attack Success Rate", self.shared_dict['att_succ_rate_writer'].get(),
                                          current_eps)
                        # write Ratio of Mission Completion
                        writer.add_scalar("Ratio of Mission Completion", self.shared_dict['scan_percent_writer'].get(),
                                          current_eps)
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
                        writer.add_scalar("Number of Cell to Scan",
                                          self.env.system.map_cell_number * self.env.system.map_cell_number,
                                          current_eps)
                        writer.add_scalar("Average Active MD+HD Number",
                                          self.shared_dict['MDHD_active_num_writer'].get(), current_eps)
                        writer.add_scalar("Average Connect_RLD MD+HD Number",
                                          self.shared_dict['MDHD_connect_RLD_num_writer'].get(), current_eps)
                        writer.add_scalar("Average Mission Complete Rate",
                                          self.shared_dict['mission_complete_rate_writer'].get(), current_eps)
                        writer.add_scalar("Average Remaining Time Writer",
                                          self.shared_dict['remaining_time_writer'].get(), current_eps)
                        writer.add_scalar("Average Energy HD", self.shared_dict['energy_HD_writer'].get(), current_eps)
                        writer.add_scalar("Average Energy MD", self.shared_dict['energy_MD_writer'].get(), current_eps)
                        writer.add_scalar("Number of init MD", self.env.system.num_MD, current_eps)
                        writer.add_scalar("Number of init HD", self.env.system.num_HD, current_eps)
                        writer.add_scalar("Attacker's max_att_budget", self.env.attacker.max_att_budget, current_eps)
                        # write recorded_max_RLD_down_time
                        writer.add_scalar("Recorded Max RLD Down Time", self.shared_dict['recorded_max_RLD_down_time_writer'].get(), current_eps)
                        # write alive_MD_num
                        writer.add_scalar("Remaining Alive MD Number", self.shared_dict['alive_MD_num_writer'].get(), current_eps)
                        # write alive_HD_num
                        writer.add_scalar("Remaining Alive HD Number", self.shared_dict['alive_HD_num_writer'].get(), current_eps)

                    ep_counter += 1

            if not self.exist_model:
                self.scheduler_def.step()  # update learning rate each episode (each agent will update lr independently)
                self.lnet_def.epsilon_decay_step()  # update epsilon each episode
                if self.is_custom_env:
                    self.scheduler_att.step()  # update learning rate each episode (each agent will update lr independently)
                    self.lnet_att.epsilon_decay_step()  # update epsilon each episode
            print(self.name, 'episode ', self.g_ep.value, 'defender reward %.1f' % score_def,
                  'attacker reward %.1f' % score_att)
        # self.res_queue.put(None)

        if self.is_custom_env: self.env.close_env()

        if writer is not None:
            writer.flush()
            writer.close()  # close SummaryWriter of TensorBoard
        return
