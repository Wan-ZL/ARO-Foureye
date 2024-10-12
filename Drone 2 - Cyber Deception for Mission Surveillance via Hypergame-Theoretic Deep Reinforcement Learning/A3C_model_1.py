'''
Project     ：Drone-DRL-HT 
File        ：A3C_model_1.py
Author      ：Zelin Wan
Date        ：2/8/23
Description : 
'''
import copy
import random
import threading
import torch
import torch.nn as nn
from utils import v_wrap, set_init, push_and_pull_with_experience_replay, record
import torch.nn.functional as F
import torch.multiprocessing as mp
import multiprocessing
import gym
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR, ExponentialLR, ReduceLROnPlateau, OneCycleLR
from Gym_Defender_and_Attacker import HyperGameSim
# from HT_attacker import HypergameTheoryAttacker
# from HT_defender import HypergameTheoryDefender
from collections import defaultdict
from datetime import datetime
# below import is for MobiHoc workshop paper. (Game Theory)
from GT_attacker import GameTheoryAttacker
from GT_defender import GameTheoryDefender


class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions, epsilon=1.0, epsilon_decay=0.9, pi_net_struc=None, v_net_struct=None,
                 HEU_prob_distribution=None, HEU_guilded=0.0, DSmT_entropy_threshold=0.003):
        super(ActorCritic, self).__init__()
        self.enable_bias = False
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        if pi_net_struc is None:
            pi_net_struc = [128]  # default 64, 64 hidden layers if not specify
        else:
            self.pi_net_struc = pi_net_struc
        if v_net_struct is None:
            v_net_struct = [128]  # default 64, 64 hidden layers if not specify
        else:
            self.v_net_struct = v_net_struct


        self.HEU_prob_distribution = HEU_prob_distribution
        self.HEU_guilded = HEU_guilded  # Whether to use HEU_guilded action or random action

        self.pi_net = self.build_Net(input_dims, n_actions, pi_net_struc)
        self.v_net = self.build_Net(input_dims, 1, v_net_struct)

        self.pi_net.apply(self.init_weight_bias)  # initial weight (normal distribution)
        self.v_net.apply(self.init_weight_bias)  # initial weight (normal distribution)
        self.predefine_weight(self.pi_net, None)

        self.distribution = torch.distributions.Categorical
        self.HEU_decision_counter = 0
        self.random_decision_counter = 0
        self.DRL_decision_counter = 0
        self.mem_buffer_size = 10000 #1000 #10000
        self.beta = 0.1 # beta for adjusting the learning rate when using prioritized experience replay.
        self.beta_increment_per_sampling = 0.0001 # beta increment per sampling
        self.mem_buffer_counter = 0
        self.mem_buffer_s = np.zeros((self.mem_buffer_size, self.input_dims))
        self.mem_buffer_a = np.zeros((self.mem_buffer_size))
        self.mem_buffer_v_target = np.zeros((self.mem_buffer_size))
        self.mem_td_error = np.zeros((self.mem_buffer_size))
        self.mini_batch_size = 32
        self.temp_buffer_size = 128 #32 #5 # 128  # temporary buffer size for collecting observations. (recommend to have this value smaller than 'mem_buffer_size')
        self.train_N_times = 5 # train N times for each sampling (the temp_buffer_size) from the memory buffer
        self.max_grad_norm = 0.5  # gradient clipping
        self.DSmT_entropy = 1 # DSmT entropy
        self.DSmT_entropy_threshold = DSmT_entropy_threshold #0.1 # DSmT entropy threshold

    def epsilon_decay_step(self):
        new_epsilon = self.epsilon * self.epsilon_decay
        self.epsilon = new_epsilon

    def HEU_guilded_decay_step(self):
        new_HEU_guilded = self.HEU_guilded * self.epsilon_decay
        self.HEU_guilded = new_HEU_guilded


    def init_weight_bias(self, layer):
        if type(layer) == nn.Linear:
            # nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain('tanh'))  # use normal distribution
            nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain('leaky_relu'))  # use normal distribution
            # nn.init.kaiming_normal_(layer.weight)
            # nn.init.normal_(layer.weight, mean=0., std=0.1)
            if self.enable_bias:
                nn.init.normal_(layer.bias, std=1 / layer.in_features)
            # nn.init.constant_(layer.bias, 0.)

    def predefine_weight(self, network, distribution):
        target_layer = len(self.pi_net_struc) + 2
        layer_count = 0
        for layer in network:
            if type(layer) == nn.Linear:
                layer_count += 1
                if layer_count == target_layer:
                    if self.HEU_prob_distribution is not None:
                        HEU_prob_pytorch = torch.tensor(self.HEU_prob_distribution, dtype=torch.float32)
                    else:
                        HEU_prob_pytorch = torch.tensor([1.0 / self.n_actions] * self.n_actions, dtype=torch.float32)

                    HEU_prob_pytorch = HEU_prob_pytorch * len(HEU_prob_pytorch)
                    print("HEU_prob_pytorch", HEU_prob_pytorch, sum(HEU_prob_pytorch))
                    HEU_diagonal_matrix = torch.diag(HEU_prob_pytorch)

                    layer.weight.data = HEU_diagonal_matrix
                    # TODO: test this on normal distribution and HEU distribution

    class PrintLayer(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            # Do your print / debug stuff here
            print("layer data", x)
            return x

    def build_Net(self, obser_space, action_space, net_struc):
        layers = []

        in_features = obser_space
        for node_num in net_struc:
            layers.append(nn.Linear(in_features, node_num, bias=self.enable_bias))

            # activation function
            # layers.append(nn.Tanh())
            layers.append(nn.LeakyReLU())

            # layer normalization
            layers.append(nn.LayerNorm(node_num))
            # layers.append(nn.BatchNorm1d(node_num))

            # dropout
            layers.append(nn.Dropout(p=0.01))

            in_features = node_num
        layers.append(nn.Linear(in_features, action_space, bias=self.enable_bias))
        # layer for debugging
        # layers.append(self.PrintLayer())
        # extra layer for mapping the output to HEU_prob_distribution
        layers.append(nn.Linear(action_space, action_space, bias=self.enable_bias))
        net = nn.Sequential(*layers)
        return net

    def HEU_prob_choose_action(self):
        self.HEU_decision_counter += 1
        return np.int64(np.random.choice(self.n_actions, p=self.HEU_prob_distribution))

    def choose_action(self, obs):
        self.eval() # set to evaluation mode. This turn off batch normalization and dropout layers.

        # # fill memory buffer
        # if self.mem_buffer_counter < self.mem_buffer_size:
        #     if self.HEU_guilded:
        #         return self.HEU_prob_choose_action()
        #         # self.HEU_decision_counter += 1
        #         # return np.int64(np.random.choice(self.n_actions, p=self.HEU_prob_distribution))
        #     else:
        #         self.random_decision_counter += 1
        #         return np.int64(np.random.choice(self.n_actions))

        # calculate the entropy of the current policy
        logits = self.pi_net(obs)
        prob = F.softmax(logits, dim=1).data
        self.DSmT_entropy = self.get_DSmT_entropy(prob.squeeze())

        # update HEU_guilded and epsilon
        # if self.DSmT_entropy <= self.DSmT_entropy_threshold and self.HEU_guilded:    # threshold for the entropy
        #     self.HEU_guilded = 0.0

        # HEU_guilded
        # if torch.rand((1,)).item() < self.HEU_guilded:
        #     return self.HEU_prob_choose_action()

        # random action
        if torch.rand((1,)).item() < self.epsilon:
            self.random_decision_counter += 1
            # return np.int64(torch.randint(0, self.n_actions, (1,)).item())
            return np.int64(np.random.choice(self.n_actions))

        # DRL action
        self.DRL_decision_counter += 1
        m = self.distribution(prob)
        final_action = m.sample().numpy()[0]
        return final_action

    def get_DSmT_entropy(self, prob):
        # prob must be one dimension numpy array

        # check prob type
        if type(prob) != np.ndarray:
            prob = np.array(prob)

        prob_len = len(prob)
        H_max = (1/prob_len) * np.log2(1/prob_len) * prob_len
        numerator = 0
        for theta in prob:
            numerator += theta * np.log2(theta)
        return numerator / H_max


    def loss_func(self, s, a, v_t, lr_factor=None):
        '''
        Calculate actor loss, critic loss and total loss
        :param s: old state
        :param a: action
        :param v_t: target value
        :return:
        '''
        if lr_factor is None:
            lr_factor = np.ones((self.mini_batch_size))

        self.train()
        logits = self.pi_net(s)
        values = self.v_net(s)
        values = values.squeeze()
        td_error = v_t - values  # temporal difference. It is also the advantage. (Refers to the 4.2 of https://www.youtube.com/watch?v=O5BlozCJBSE&t=378s)
        c_loss = td_error.pow(2)  # critic loss (mean square error)

        probs = F.softmax(logits, dim=1)  # use softmax to normalize the logits to probability
        m = self.distribution(probs)  # use Categorical to generate a distribution

        # This step refers to the line 123 of https://github.com/colinskow/move37/blob/master/actor_critic/a3c.py
        exp_v = m.log_prob(a) * td_error.detach().squeeze()
        a_loss = -exp_v
        total_loss = c_loss + a_loss
        total_loss_mean = total_loss.mean()
        total_loss_weighted = total_loss * v_wrap(lr_factor)
        total_loss_weighted_mean = total_loss_weighted.mean()
        c_loss_mean = c_loss.mean()
        a_loss_mean = a_loss.mean()
        return total_loss_mean, c_loss_mean, a_loss_mean, total_loss_weighted_mean, td_error


class Agent(mp.Process):
    def __init__(self, gnet_def, gnet_att, opt_def, opt_att, shared_dict, config_def, config_att, def_attri_dict,
                 att_attri_dict, fixed_seed, trial, name_id: int, def_select_method='fixed', att_select_method='fixed',
                 is_custom_env=True,
                 miss_dur=150, target_size=5, max_att_budget=5, num_HD=5, DSmT_entropy_threshold=0.003, epsilon=1.0):
        super(Agent, self).__init__()
        self.name_id = name_id
        self.name = 'w%02i' % name_id
        # self.exp_scheme = exp_scheme
        self.def_select_method = def_select_method
        self.att_select_method = att_select_method
        self.scheme_name = self.att_select_method + '-' + self.def_select_method
        self.gnet_def = gnet_def
        self.gnet_att = gnet_att
        self.opt_def = opt_def
        self.opt_att = opt_att
        self.shared_dict = shared_dict
        self.g_ep = shared_dict['global_ep']
        self.g_r_list = shared_dict['glob_r_list']
        self.lr_decay_def = config_def["LR_decay"]
        self.lr_decay_att = config_att["LR_decay"]
        self.is_custom_env = is_custom_env
        self.miss_dur = miss_dur
        self.target_size = target_size
        self.max_att_budget = max_att_budget
        self.num_HD = num_HD
        if is_custom_env:
            self.env = HyperGameSim(fixed_seed=fixed_seed, miss_dur=miss_dur, target_size=target_size,
                                    max_att_budget=max_att_budget, num_HD=num_HD, def_select_method=def_select_method)
            self.input_dims_def = self.env.observation_space['def'].shape[0]
            self.n_actions_def = self.env.action_space['def'].n
            self.input_dims_att = self.env.observation_space['att'].shape[0]
            self.n_actions_att = self.env.action_space['att'].n
            # update setting for sensitivity analysis
        else:
            self.env = gym.make('CartPole-v1')
            self.input_dims_def = self.env.observation_space.shape[0]
            self.n_actions_def = self.env.action_space.n
            self.input_dims_att = 1
            self.n_actions_att = 1

        # defender's local AC
        self.lnet_def = ActorCritic(input_dims=self.input_dims_def, n_actions=self.n_actions_def,
                                    epsilon=config_def["epsilon"],
                                    epsilon_decay=config_def["epsilon_decay"], pi_net_struc=gnet_def.pi_net_struc,
                                    v_net_struct=gnet_def.v_net_struct,
                                    HEU_prob_distribution=gnet_def.HEU_prob_distribution,
                                    HEU_guilded=gnet_def.HEU_guilded,
                                    DSmT_entropy_threshold=DSmT_entropy_threshold)  # local network for defender

        # attacker's local AC
        self.lnet_att = ActorCritic(input_dims=self.input_dims_att, n_actions=self.n_actions_att,
                                    epsilon=config_att["epsilon"],
                                    epsilon_decay=config_att["epsilon_decay"], pi_net_struc=gnet_att.pi_net_struc,
                                    v_net_struct=gnet_att.v_net_struct,
                                    HEU_prob_distribution=gnet_att.HEU_prob_distribution,
                                    HEU_guilded=gnet_att.HEU_guilded,
                                    DSmT_entropy_threshold=DSmT_entropy_threshold)  # local network for attacker

        self.gamma_def = config_def["gamma"]
        self.gamma_att = config_att["gamma"]
        self.MAX_EP = config_def["glob_episode_thred"]  # defender and attacker should have the same
        # 'glob_episode_thred', so I didn't make them different here
        self.trial = trial
        # scheduler for learning rate decay
        # self.scheduler_def = ExponentialLR(self.opt_def, gamma=self.lr_decay_def)
        # self.scheduler_att = ExponentialLR(self.opt_att, gamma=self.lr_decay_att)
        self.scheduler_def = ReduceLROnPlateau(self.opt_def, mode='max', factor=self.lr_decay_def, patience=10)
        self.scheduler_att = ReduceLROnPlateau(self.opt_att, mode='max', factor=self.lr_decay_att, patience=10)
        self.transaction_dict = {}
        # HEU agent
        self.def_HEU_agent = self.create_HEU_agents(def_attri_dict,
                                                    'def')  # create defender's HEU agent to provide reward
        self.att_HEU_agent = self.create_HEU_agents(att_attri_dict,
                                                    'att')  # create attacker's HEU agent to provide reward

        self.DSmT_entropy_threshold = DSmT_entropy_threshold
        self.epsilon = epsilon

    def create_HEU_agents(self, attribute_dict, agent_type='def'):
        '''
        create HEU agent according to the agent_type and update its attribute
        :param attribute_dict: a dictionary contains the attribute of the HEU agent
        :param agent_type: 'def' or 'att'
        :return:
        '''
        # deep copy the attribute_dict to avoid changing the original attribute_dict
        attribute_dict = copy.deepcopy(attribute_dict)

        # create HEU agent
        if agent_type == 'def':
            # create defender's HEU agent
            agent = GameTheoryDefender(self.env)
            # update defender's HEU agent's attribute
            if self.def_select_method != 'HEU':
                for key, value in attribute_dict.items():
                    setattr(agent, key, value)
                # keep HEU_prob_record as 2D array
                agent.HEU_prob_record = agent.HEU_prob_record[np.newaxis]
        elif agent_type == 'att':
            # create attacker's HEU agent
            agent = GameTheoryAttacker(self.env)
            # update attacker's HEU agent's attribute
            if self.att_select_method != 'HEU':
                for key, value in attribute_dict.items():
                    setattr(agent, key, value)
                # keep HEU_prob_record as 2D array
                agent.HEU_prob_record = agent.HEU_prob_record[np.newaxis]
        else:
            raise ValueError("agent_type should be 'def' or 'att'")

        # keep action_space as int
        agent.action_space = int(agent.action_space)

        return agent

    # def lambda_function_def(self, epoch):  # epoch increase one when scheduler_def.step() is called
    #     return self.lr_decay_def ** epoch
    #
    # def lambda_function_att(self, epoch):  # epoch increase one when scheduler_att.step() is called
    #     return self.lr_decay_att ** epoch

    def run(self):
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
                path = "/projects/zelin1/Drone_DRL_HT/data/" + str(self.miss_dur) + "_" + str(self.max_att_budget) + "_" + \
                       str(self.num_HD) + "/"
            else:
                # set path for my laptop
                path = "data/" + str(self.miss_dur) + "_" + str(self.max_att_budget) + "_" + str(self.num_HD) + "/"

            current_time = datetime.now().strftime("%Y-%m-%d-%H_%M_%S-%f")
            writer = SummaryWriter(log_dir=path + "scheme_" + self.scheme_name + "/each_run-"
                                           + "scheme_" + self.scheme_name + "-Time_" + current_time
                                           + "-Trial_" + trial_num_str + "-eps")
        else:
            writer = None

        ep_counter = 0
        accum_step_counter = 0
        reward_def_dict = defaultdict(list)
        reward_att_dict = defaultdict(list)
        score_def_dict = defaultdict(list)
        score_att_dict = defaultdict(list)

        while self.g_ep.value < self.MAX_EP:
            # run until episode limit reached
            # print("Number of active threads:", threading.active_count())
            with self.g_ep.get_lock():
                temp_ep = self.g_ep.value

            if self.is_custom_env:
                obs_out = self.env.reset()
                obs_def = obs_out['def']
                obs_att = obs_out['att']
            else:
                obs_def, _ = self.env.reset()

            buffer_s_def, buffer_a_def, buffer_r_def = [], [], []
            buffer_s_att, buffer_a_att, buffer_r_att = [], [], []
            total_loss_set_def = []
            c_loss_set_def = []
            a_loss_set_def = []
            total_loss_set_att = []
            c_loss_set_att = []
            a_loss_set_att = []
            att_counter_set = []
            att_succ_rate_set = []
            MDHD_active_set = []
            MDHD_connect_RLD_set = []
            remaining_time_set = []
            energy_HD_set = []
            energy_MD_set = []
            def_stra_counter = np.zeros(10)
            att_stra_counter = np.zeros(10)
            score_att = 0
            score_def = 0
            step_counter = 0
            done = False

            while not done:
                # run until environment end

                # choose action
                action_def = None
                action_att = None
                if self.is_custom_env:
                    if self.def_select_method in ['DRL', 'HT-DRL']:
                        action_def = self.lnet_def.choose_action(v_wrap(obs_def[None, :]))
                    elif self.def_select_method == 'HEU':
                        # action_def = self.lnet_def.HEU_prob_choose_action() # TODO: after 100 test, decide whether to use probability distribution.
                        action_def = np.int64(self.def_HEU_agent.act())
                    else:
                        pass

                    if self.att_select_method in ['DRL', 'HT-DRL']:
                        action_att = self.lnet_att.choose_action(v_wrap(obs_att[None, :]))
                    elif self.att_select_method == 'HEU':
                        # action_att = self.lnet_att.HEU_prob_choose_action()
                        action_att = np.int64(self.att_HEU_agent.act())
                    else:
                        pass

                    # if self.exp_scheme == 0:  # A-random D-a3c
                    #     action_def = self.lnet_def.choose_action(v_wrap(obs_def[None, :]))
                    # elif self.exp_scheme == 1:  # A-a3c D-random
                    #     action_att = self.lnet_att.choose_action(v_wrap(obs_att[None, :]))
                    # elif self.exp_scheme == 2:  # A-a3c D-a3c
                    #     action_def = self.lnet_def.choose_action(v_wrap(obs_def[None, :]))
                    #     action_att = self.lnet_att.choose_action(v_wrap(obs_att[None, :]))
                    # else:  # A-random D-random
                    #     pass
                else:
                    action_def = self.lnet_def.choose_action(v_wrap(obs_def[None, :]))

                if self.is_custom_env:  # for Drone environment
                    # interaction with environment
                    the_obs_, the_reward, done, info = self.env.step(action_def=action_def, action_att=action_att)

                    action_def = info["action_def"]
                    action_att = info["action_att"]

                    # replace the reward to the HEU reward
                    # defender's HEU
                    # self.def_HEU_agent.self_strategy = action_def
                    # self.def_HEU_agent.refresh_HEU_for_each_strategy()
                    # reward_def = self.def_HEU_agent.hypergame_expected_utility[action_def]
                    # self.def_HEU_agent.observe(action_att)
                    # attacker's HEU
                    # self.att_HEU_agent.self_strategy = action_att
                    # self.att_HEU_agent.refresh_HEU_for_each_strategy()
                    # reward_att = self.att_HEU_agent.hypergame_expected_utility[action_att]
                    # self.att_HEU_agent.observe(action_def)

                    # add the observation HEU elements to the observation
                    obs_new_def = the_obs_['def']
                    self.def_HEU_agent.observe(action_att)
                    # print("HEU observation:", self.def_HEU_agent.Observation_record_counter)
                    obs_new_att = the_obs_['att']
                    self.att_HEU_agent.observe(action_def)

                    reward_def = the_reward['def']
                    reward_att = the_reward['att']


                    # add data from each step (for custom env only)
                    def_stra_counter[action_def] += 1
                    att_stra_counter[action_att] += 1
                    MDHD_active_set.append(info["MDHD_active_num"])
                    MDHD_connect_RLD_set.append(info["MDHD_connected_num"])
                    remaining_time_set.append(info["remaining_time"])
                    energy_HD_set.append(info["energy_HD"])
                    energy_MD_set.append(info["energy_MD"])
                else:
                    obs_new_def, reward_def, done, truncated, info = self.env.step(
                        action_def)  # assign reward to reward_def, ignore reward_att
                    # self.env.render()
                    reward_att = 0
                score_def += reward_def
                score_att += reward_att

                reward_def_dict[step_counter].append(reward_def)
                reward_att_dict[step_counter].append(reward_att)
                score_def_dict[step_counter].append(score_def)
                score_att_dict[step_counter].append(score_att)


                # Check if same state and action pair gives the same reward
                # is_defender = False
                # # obs_new_def = np.round(obs_new_def, 2)
                # if is_defender:
                #     transaction = list(obs_def) + [action_def]
                #     transaction = tuple(transaction)
                #     if transaction in self.transaction_dict:
                #         if self.transaction_dict[transaction] != reward_def:
                #             print("Error: Same state and action pair gives different reward")
                #             print("State: ", obs_def)
                #             print("Action: ", action_def)
                #             print("Reward: ", reward_def)
                #             print("Previous reward: ", self.transaction_dict[transaction])
                #         else:
                #             print("Defender: Same state and action pair gives same reward", reward_def)
                #     else:
                #         self.transaction_dict[transaction] = reward_def
                #         # print("transaction:", transaction)
                #         # print("transaction_dict size:", len(self.transaction_dict))
                # else:
                #     transaction = obs_att.tolist() + [action_att]
                #     transaction = tuple(transaction)
                #     if transaction in self.transaction_dict:
                #         if self.transaction_dict[transaction] != reward_att:
                #             print("Error: Same state and action pair gives different reward")
                #             print("State: ", obs_att)
                #             print("Action: ", action_att)
                #             print("Reward: ", reward_att)
                #             print("Previous reward: ", self.transaction_dict[transaction])
                #         else:
                #             print("Attacker: Same state and action pair gives same reward", reward_att)
                #     else:
                #         self.transaction_dict[transaction] = reward_att
                #         # print("transaction:", transaction)
                #         # print("transaction_dict size:", len(self.transaction_dict))





                # add to memory
                buffer_a_def.append(action_def)
                buffer_s_def.append(obs_def)
                buffer_r_def.append(reward_def)
                if self.is_custom_env:
                    buffer_a_att.append(action_att)
                    buffer_s_att.append(obs_att)
                    buffer_r_att.append(reward_att)

                # update global and assign to local net
                if step_counter % self.lnet_def.temp_buffer_size == 0 or done:
                    # sync
                    total_loss_def, c_loss_def, a_loss_def = push_and_pull_with_experience_replay(self.opt_def,
                                                                                                  self.lnet_def,
                                                                                                  self.gnet_def, done,
                                                                                                  obs_new_def,
                                                                                                  buffer_s_def,
                                                                                                  buffer_a_def,
                                                                                                  buffer_r_def,
                                                                                                  self.gamma_def)
                    if self.is_custom_env:
                        total_loss_att, c_loss_att, a_loss_att = push_and_pull_with_experience_replay(self.opt_att,
                                                                                                      self.lnet_att,
                                                                                                      self.gnet_att,
                                                                                                      done, obs_new_att,
                                                                                                      buffer_s_att,
                                                                                                      buffer_a_att,
                                                                                                      buffer_r_att,
                                                                                                      self.gamma_att)

                    # save loss data for display
                    total_loss_set_def.append(total_loss_def)
                    c_loss_set_def.append(c_loss_def)
                    a_loss_set_def.append(a_loss_def)
                    if self.is_custom_env:
                        total_loss_set_att.append(total_loss_att)
                        c_loss_set_att.append(c_loss_att)
                        a_loss_set_att.append(a_loss_att)

                    # clear memory
                    buffer_s_def, buffer_a_def, buffer_r_def = [], [], []
                    if self.is_custom_env:
                        buffer_s_att, buffer_a_att, buffer_r_att = [], [], []

                # add per step data to writer
                # if writer is not None:
                #     writer.add_scalar("Immediate Reward/Defender (x-axis is step accumulated)", reward_def, accum_step_counter)
                #     writer.add_scalar("Immediate Reward/Attacker (x-axis is step accumulated)", reward_att, accum_step_counter)
                #     writer.add_scalar("Accumulated Reward/Defender (x-axis is step accumulated)", score_def, accum_step_counter)
                #     writer.add_scalar("Accumulated Reward/Attacker (x-axis is step accumulated)", score_att, accum_step_counter)
                #     writer.add_scalar("Entropy/Defender (x-axis is step accumulated)", self.lnet_def.DSmT_entropy, accum_step_counter)
                #     writer.add_scalar("Entropy/Attacker (x-axis is step accumulated)", self.lnet_att.DSmT_entropy, accum_step_counter)

                # update obs
                obs_def = obs_new_def
                if self.is_custom_env:
                    obs_att = obs_new_att
                step_counter += 1
                accum_step_counter += 1

            score_def_avg = score_def / step_counter if step_counter != 0 else 0
            score_att_avg = score_att / step_counter if step_counter != 0 else 0
            # save data to shared dictionary for tensorboard
            with self.g_ep.get_lock():
                self.g_ep.value += 1
                self.shared_dict['eps_writer'].put(temp_ep)
                if self.def_select_method in ['HEU', 'DRL', 'HT-DRL']:
                    # defender's reward to guid optuna
                    self.g_r_list.append(score_def)
                else:
                    # uses attacker's reward to guid optuna
                    self.g_r_list.append(score_att)

                # if self.exp_scheme == 0 or self.exp_scheme == 2:
                    # 'A-random D-a3c' and 'A-a3c D-a3c' use defender's reward to guid optuna
                    # self.g_r_list.append(score_def)
                # elif self.exp_scheme == 1:
                    # 'A-a3c D-random' uses attacker's reward to guid optuna
                    # self.g_r_list.append(score_att)
                # else:  # 'A-random D-random' does need optuna
                #     pass

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
                # save decision ratio. Add 1 to avoid zero division
                def_decision_total = float(
                    self.lnet_def.HEU_decision_counter + self.lnet_def.random_decision_counter + self.lnet_def.DRL_decision_counter) + 1
                att_decision_total = float(
                    self.lnet_att.HEU_decision_counter + self.lnet_att.random_decision_counter + self.lnet_att.DRL_decision_counter) + 1
                self.shared_dict['def_HEU_decision_ratio'].put(self.lnet_def.HEU_decision_counter / def_decision_total)
                self.shared_dict['att_HEU_decision_ratio'].put(self.lnet_att.HEU_decision_counter / att_decision_total)
                self.shared_dict['def_random_decision_ratio'].put(
                    self.lnet_def.random_decision_counter / def_decision_total)
                self.shared_dict['att_random_decision_ratio'].put(
                    self.lnet_att.random_decision_counter / att_decision_total)
                self.shared_dict['def_DRL_decision_ratio'].put(self.lnet_def.DRL_decision_counter / def_decision_total)
                self.shared_dict['att_DRL_decision_ratio'].put(self.lnet_att.DRL_decision_counter / att_decision_total)
                # reset decision counter
                self.lnet_def.HEU_decision_counter = 0
                self.lnet_def.random_decision_counter = 0
                self.lnet_def.DRL_decision_counter = 0
                self.lnet_att.HEU_decision_counter = 0
                self.lnet_att.random_decision_counter = 0
                self.lnet_att.DRL_decision_counter = 0
                if self.is_custom_env:
                    # self.shared_dict['mission_condition_writer'].put(info["mission_condition"])
                    self.shared_dict['total_energy_consump_writer'].put(info["total_energy_consump"])
                    self.shared_dict['scan_percent_writer'].put(info["scan_percent"])
                    self.shared_dict['MDHD_active_num_writer'].put(
                        sum(MDHD_active_set) / len(MDHD_active_set) if len(MDHD_active_set) else 0)
                    self.shared_dict['MDHD_connect_RLD_num_writer'].put(
                        sum(MDHD_connect_RLD_set) / len(MDHD_connect_RLD_set) if len(MDHD_connect_RLD_set) else 0)
                    self.shared_dict['remaining_time_writer'].put(
                        sum(remaining_time_set) / len(remaining_time_set) if len(remaining_time_set) else 0)
                    self.shared_dict['energy_HD_writer'].put(
                        sum(energy_HD_set) / len(energy_HD_set) if len(energy_HD_set) else 0)
                    self.shared_dict['energy_MD_writer'].put(
                        sum(energy_MD_set) / len(energy_MD_set) if len(energy_MD_set) else 0)

            # ==== tensorboard writer ====
            # Only agent (index 0) can write to tensorboard
            if writer is not None:
                while not self.shared_dict['eps_writer'].empty():
                    # use episode as index of tensorboard
                    current_eps = self.shared_dict['eps_writer'].get()
                    current_eps = ep_counter
                    # write score
                    writer.add_scalar("Accumulated Reward/Defender", self.shared_dict["score_def_writer"].get(),
                                      current_eps)
                    writer.add_scalar("Accumulated Reward/Attacker", self.shared_dict["score_att_writer"].get(),
                                      current_eps)
                    writer.add_scalar("Averaged Reward/Defender", self.shared_dict["score_def_avg_writer"].get(),
                                      current_eps)
                    writer.add_scalar("Averaged Reward/Attacker", self.shared_dict["score_att_avg_writer"].get(),
                                      current_eps)
                    # write lr
                    writer.add_scalar("Learning rate/(def)", self.shared_dict["lr_def_writer"].get(), current_eps)
                    writer.add_scalar("Learning rate/(att)", self.shared_dict["lr_att_writer"].get(), current_eps)
                    # write epsilon
                    writer.add_scalar("Epsilon (random action probability)/(def)",
                                      self.shared_dict["epsilon_def_writer"].get(),
                                      current_eps)
                    writer.add_scalar("Epsilon (random action probability)/(att)",
                                      self.shared_dict["epsilon_att_writer"].get(),
                                      current_eps)
                    # write mission time (step)
                    writer.add_scalar("Mission Time (step)", step_counter, current_eps)
                    # write loss
                    writer.add_scalar("Defender's Loss/Total Loss", self.shared_dict['t_loss_def_writer'].get(), current_eps)
                    writer.add_scalar("Defender's Loss/Critic Loss", self.shared_dict['c_loss_def_writer'].get(),
                                      current_eps)
                    writer.add_scalar("Defender's Loss/Actor Loss", self.shared_dict['a_loss_def_writer'].get(), current_eps)
                    writer.add_scalar("Attacker's Loss/Total Loss", self.shared_dict['t_loss_att_writer'].get(), current_eps)
                    writer.add_scalar("Attacker's Loss/Critic Loss", self.shared_dict['c_loss_att_writer'].get(),
                                      current_eps)
                    writer.add_scalar("Attacker's Loss/Actor Loss", self.shared_dict['a_loss_att_writer'].get(), current_eps)
                    # write strategy counter
                    # for defender
                    def_stra_counter = self.shared_dict['def_stra_count_writer'].get()
                    count_sum = np.sum(def_stra_counter)
                    def_stra_counter = def_stra_counter / count_sum if count_sum != 0 else def_stra_counter
                    for i in range(10):
                        writer.add_scalar("Defender Strategy/(" + str(i) + ") Freq.", def_stra_counter[i], current_eps)
                    # for attacker
                    att_stra_counter = self.shared_dict['att_stra_count_writer'].get()
                    count_sum = np.sum(att_stra_counter)
                    att_stra_counter = att_stra_counter / count_sum if count_sum != 0 else att_stra_counter
                    for i in range(10):
                        writer.add_scalar("Attacker Strategy/(" + str(i) + ") Freq.", att_stra_counter[i], current_eps)
                    # write HEU decision ratio
                    writer.add_scalar("Defender Decision Ratio/HEU", self.shared_dict['def_HEU_decision_ratio'].get(),
                                      current_eps)
                    writer.add_scalar("Attacker Decision Ratio/HEU", self.shared_dict['att_HEU_decision_ratio'].get(),
                                      current_eps)
                    # write random decision ratio
                    writer.add_scalar("Defender Decision Ratio/Random",
                                      self.shared_dict['def_random_decision_ratio'].get(), current_eps)
                    writer.add_scalar("Attacker Decision Ratio/Random",
                                      self.shared_dict['att_random_decision_ratio'].get(), current_eps)
                    # write DRL decision ratio
                    writer.add_scalar("Defender Decision Ratio/DRL", self.shared_dict['def_DRL_decision_ratio'].get(),
                                      current_eps)
                    writer.add_scalar("Attacker Decision Ratio/DRL", self.shared_dict['att_DRL_decision_ratio'].get(),
                                      current_eps)
                    if self.is_custom_env:
                        # write Ratio of Mission Completion
                        writer.add_scalar("Ratio of Mission Completion", self.shared_dict['scan_percent_writer'].get(),
                                          current_eps)
                        # write energy cunsumption for all drone (added them together)
                        writer.add_scalar("Energy/Energy Consumption", self.shared_dict['total_energy_consump_writer'].get(),
                                          current_eps)
                        # write other metrics
                        writer.add_scalar("Settings/Max Mission Duration", self.env.system.mission_duration_max, current_eps)
                        writer.add_scalar("Settings/Number of Cell to Scan",
                                          self.env.system.map_cell_number * self.env.system.map_cell_number,
                                          current_eps)
                        writer.add_scalar("Drone Number/Average Active MD+HD Number",
                                          self.shared_dict['MDHD_active_num_writer'].get(), current_eps)
                        writer.add_scalar("Drone Number/Average Connect_RLD MD+HD Number",
                                          self.shared_dict['MDHD_connect_RLD_num_writer'].get(), current_eps)
                        writer.add_scalar("Average Remaining Time",
                                          self.shared_dict['remaining_time_writer'].get(), current_eps)
                        writer.add_scalar("Energy/Average Energy HD", self.shared_dict['energy_HD_writer'].get(), current_eps)
                        writer.add_scalar("Energy/Average Energy MD", self.shared_dict['energy_MD_writer'].get(), current_eps)
                        writer.add_scalar("Settings/Number of init MD", self.env.system.num_MD, current_eps)
                        writer.add_scalar("Settings/Number of init HD", self.env.system.num_HD, current_eps)
                        writer.add_scalar("Settings/Attacker's max_att_budget", self.env.attacker.max_att_budget, current_eps)
                        writer.add_scalar("Settings/Entropy Threshold", self.lnet_def.DSmT_entropy_threshold, current_eps)
                    ep_counter += 1

            if writer is not None:
                writer.flush()


            # only the first thread (w00) can update the lr and epsilon
            if self.lnet_def.mem_buffer_counter > self.lnet_def.mem_buffer_size:
                # only change the learning rate and epsilon when the memory buffer is full
                self.scheduler_def.step(score_def)  # update learning rate each episode (each agent will update lr independently)
                self.lnet_def.HEU_guilded_decay_step()  # update HEU_guilded ratio each episode
                if not self.lnet_def.HEU_guilded:  # if using HEU, then no need to decay epsilon
                    self.lnet_def.epsilon_decay_step()  # update epsilon each episode

                if self.is_custom_env:
                    if self.name_id == 0:
                        self.scheduler_att.step(score_att)  # update learning rate each episode (each agent will update lr independently)
                        self.lnet_att.HEU_guilded_decay_step()  # update HEU_guilded ratio each episode
                        if not self.lnet_att.HEU_guilded:  # if using HEU, then no need to decay epsilon
                            self.lnet_att.epsilon_decay_step()  # update epsilon each episode

            print(self.name, 'episode ', self.g_ep.value, 'defender reward %.1f' % score_def,
                  'attacker reward %.1f' % score_att, "epsilon: ", self.lnet_def.epsilon,
                  "learning rate: ", self.opt_def.param_groups[0]['lr'], "beta: ", self.lnet_def.beta)
            # print("self.def_HEU_agent.HEU:", self.def_HEU_agent.hypergame_expected_utility, self.def_HEU_agent.HEU_prob)

            # print("learning rate: ", self.lnet_def.opt_def.param_groups[0]['lr'])
        # self.res_queue.put(None)

        print("MAX_EP reached")
        print("ID name", self.name_id)

        if writer is not None:
            # write reward/score per step
            for step_id in range(self.env.system.mission_duration_max):
                writer.add_scalar("Immediate Reward/Defender (x-axis is step)", np.mean(reward_def_dict[step_id]), step_id)
                writer.add_scalar("Immediate Reward/Attacker (x-axis is step)", np.mean(reward_att_dict[step_id]), step_id)
                writer.add_scalar("Accumulated Reward/Defender (x-axis is step)", np.mean(score_def_dict[step_id]), step_id)
                writer.add_scalar("Accumulated Reward/Attacker (x-axis is step)", np.mean(score_att_dict[step_id]), step_id)
            # flush and close writer
            print("flush writer")
            writer.flush()
            print("close writer")
            writer.close()  # close SummaryWriter of TensorBoard

            # clean the queue to avoid deadlock in the main process when data in queue is not consumed.
            for key, val in self.shared_dict.items():
                if isinstance(self.shared_dict[key], multiprocessing.queues.Queue):
                    while not self.shared_dict[key].empty():
                        print("key", key, "cleaning", self.shared_dict[key].get())

        print("start return")
        return
