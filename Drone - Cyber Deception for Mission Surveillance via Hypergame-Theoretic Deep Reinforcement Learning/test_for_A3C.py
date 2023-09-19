# Code is heavily inspired by Morvan Zhou's code. Please check out
# his work at github.com/MorvanZhou/pytorch-A3C
import gym
import torch
import time
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.functional as F
import numpy as np
from multiprocessing import Manager

from shared_classes import SharedAdam
from utils import v_wrap_2, v_wrap, push_and_pull_with_experience_replay
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions, epsilon=0.1, epsilon_decay=0.9, pi_net_struc=None, v_net_struct=None,
                 fixed_seed=True, HEU_prob_distribution=None, HEU_guilded_ratio=0.0):
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

        if fixed_seed:
            print("fix init seed")
            torch.manual_seed(0)

        self.HEU_prob_distribution = HEU_prob_distribution
        self.HEU_guilded_ratio = HEU_guilded_ratio  # the ratio of using HEU_guilded action or random action. 0.0 means
        # all random action

        self.pi_net = self.build_Net(input_dims, n_actions, pi_net_struc)
        self.v_net = self.build_Net(input_dims, 1, v_net_struct)

        self.pi_net.apply(self.init_weight_bias)  # initial weight (normal distribution)
        self.v_net.apply(self.init_weight_bias)  # initial weight (normal distribution)

        self.distribution = torch.distributions.Categorical
        self.HEU_decision_counter = 0
        self.random_decision_counter = 0
        self.DRL_decision_counter = 0
        self.mem_buffer_size = 1000 #10000
        self.beta = 0.1 # beta for adjusting the learning rate when using prioritized experience replay.
        self.beta_increment_per_sampling = 0.001 # beta increment per sampling
        self.mem_buffer_counter = 0
        self.mem_buffer_s = np.zeros((self.mem_buffer_size, self.input_dims))
        self.mem_buffer_a = np.zeros((self.mem_buffer_size))
        self.mem_buffer_v_target = np.zeros((self.mem_buffer_size))
        self.mem_td_error = np.zeros((self.mem_buffer_size))
        self.mini_batch_size = 32 #32
        self.temp_buffer_size = 5 #128  # temporary buffer size for collecting observations. (recommend to have this value smaller than 'mem_buffer_size')
        self.train_N_times = 1  # train N times for each sampling (the temp_buffer_size) from the memory buffer
        self.max_grad_norm = 0.5  # gradient clipping

    def epsilon_decay_step(self):
        new_epsilon = self.epsilon * self.epsilon_decay
        self.epsilon = new_epsilon

    def init_weight_bias(self, layer):
        if type(layer) == nn.Linear:
            # nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain('tanh'))  # use normal distribution
            nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain('leaky_relu'))  # use normal distribution
            # nn.init.kaiming_normal_(layer.weight)
            # nn.init.normal_(layer.weight, mean=0., std=0.1)
            if self.enable_bias:
                nn.init.normal_(layer.bias, std=1 / layer.in_features)
            # nn.init.constant_(layer.bias, 0.)

    def build_Net(self, obser_space, action_space, net_struc):
        layers = []
        in_features = obser_space
        # for i in range(n_layers):
        for node_num in net_struc:
            layers.append(nn.Linear(in_features, node_num, bias=self.enable_bias))
            # layer normalization
            layers.append(nn.LayerNorm(node_num))
            # layers.append(nn.BatchNorm1d(node_num))

            # dropout
            # layers.append(nn.Dropout(p=0.1))

            # activation function
            # layers.append(nn.Tanh())
            layers.append(nn.LeakyReLU())
            in_features = node_num
        layers.append(nn.Linear(in_features, action_space))
        net = nn.Sequential(*layers)
        return net

    def choose_action(self, obs):
        self.eval() # set to evaluation mode. This turn off batch normalization and dropout layers.

        if torch.rand((1,)).item() < self.epsilon:
            # random action or HEU_guilded
            if torch.rand((1,)).item() < self.HEU_guilded_ratio:
                # HEU_guilded
                self.HEU_decision_counter += 1
                return np.int64(np.random.choice(self.n_actions, p=self.HEU_prob_distribution))
            else:
                # random action
                self.random_decision_counter += 1
                return np.int64(np.random.choice(self.n_actions))

        self.DRL_decision_counter += 1
        # print("obs: ", obs)
        # # obs + obs
        # temp_obs = v_wrap(np.concatenate((obs, obs), axis=0))
        logits = self.pi_net(obs)
        # print("logits: ", logits)
        prob = F.softmax(logits, dim=1).data

        m = self.distribution(prob)
        final_action = m.sample().numpy()[0]
        return final_action

    def loss_func(self, s, a, v_t, lr_adjuster=None):
        '''
        Calculate actor loss, critic loss and total loss
        :param s: old state
        :param a: action
        :param v_t: target value
        :return:
        '''
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
        total_loss_weighted = total_loss * v_wrap(lr_adjuster)
        total_loss_weighted_mean = total_loss_weighted.mean()
        c_loss_mean = c_loss.mean()
        a_loss_mean = a_loss.mean()
        return total_loss_mean, c_loss_mean, a_loss_mean, total_loss_weighted_mean, td_error


class Agent(mp.Process):
    def __init__(self, gnet_def, gnet_att, opt_def, opt_att, shared_dict, config_def, config_att, def_attri_dict,
                 att_attri_dict, fixed_seed, trial, name_id: int, exp_scheme=0, player='def', is_custom_env=True,
                 miss_dur=150, target_size=5, max_att_budget=5, num_HD=5):
        super(Agent, self).__init__()
        self.name_id = name_id
        self.name = 'w%02i' % name_id
        self.exp_scheme = exp_scheme
        self.gnet_def = gnet_def
        self.opt_def = opt_def
        self.shared_dict = shared_dict
        self.g_ep = shared_dict['global_ep']
        if gnet_def.HEU_guilded_ratio > 0.0:
            self.HEU_guilded = True
        else:
            self.HEU_guilded = False

        self.env = gym.make(env_id)
        self.input_dims_def = self.env.observation_space.shape[0]
        self.n_actions_def = self.env.action_space.n

        self.lnet_def = ActorCritic(input_dims=self.input_dims_def, n_actions=self.n_actions_def,
                                    epsilon=config_def["epsilon"],
                                    epsilon_decay=config_def["epsilon_decay"], pi_net_struc=gnet_def.pi_net_struc,
                                    v_net_struct=gnet_def.v_net_struct, fixed_seed=fixed_seed,
                                    HEU_prob_distribution=gnet_def.HEU_prob_distribution,
                                    HEU_guilded_ratio=gnet_def.HEU_guilded_ratio)  # local network for defender

        self.gamma_def = config_def["gamma"]
        self.MAX_EP = config_def["glob_episode_thred"]  # defender and attacker should have the same
        # 'glob_episode_thred', so I didn't make them different here
        self.player = player

    def run(self):
        while self.g_ep.value < self.MAX_EP:
            obs_def, _ = self.env.reset()

            buffer_s_def, buffer_a_def, buffer_r_def = [], [], []
            score_def = 0
            total_step = 1
            done = False

            while not done: # TODO: original code have 'score_def < 1000 and score_att < 1000'
                action_def = self.lnet_def.choose_action(v_wrap(obs_def[None, :]))
                obs_new_def, reward_def, done, truncated, info = self.env.step(action_def)
                score_def += reward_def

                buffer_a_def.append(action_def)
                buffer_s_def.append(obs_def)
                buffer_r_def.append(reward_def)

                if total_step % self.lnet_def.temp_buffer_size == 0 or done:
                    # sync
                    total_loss_def, c_loss_def, a_loss_def = push_and_pull_with_experience_replay(self.opt_def,
                                                                                                  self.lnet_def,
                                                                                                  self.gnet_def, done,
                                                                                                  obs_new_def,
                                                                                                  buffer_s_def,
                                                                                                  buffer_a_def,
                                                                                                  buffer_r_def,
                                                                                                  self.gamma_def)
                obs_def = obs_new_def
                total_step += 1

            with self.g_ep.get_lock():
                self.g_ep.value += 1

            print(self.name, 'episode ', self.g_ep.value, 'defender reward %.1f' % score_def, "epsilon: ", self.lnet_def.epsilon,
                  "learning rate: ", self.opt_def.param_groups[0]['lr'])



lr = 1e-4
env_id = 'CartPole-v1'
n_actions = 2
input_dims = [4]
N_GAMES = 3000
T_MAX = 5
fixed_seed=False

if __name__ == '__main__':
    config_def = dict(glob_episode_thred=N_GAMES, min_episode=1000, gamma=0.99,
                      lr=0.0001, LR_decay=1.0, epsilon=0.0,
                      epsilon_decay=0.99, pi_net_struc=[128, 64, 32], v_net_struct=[128, 64, 32])

    env = gym.make(env_id)
    input_dims_def = env.observation_space.shape[0]
    n_actions_def = env.action_space.n
    att_HEU_mean = None
    def_HEU_mean = None
    HEU_guilded_ratio = 0.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    glob_AC_def = ActorCritic(input_dims=input_dims_def, n_actions=n_actions_def, epsilon=config_def["epsilon"],
                              epsilon_decay=config_def["epsilon_decay"], pi_net_struc=config_def["pi_net_struc"],
                              v_net_struct=config_def["v_net_struct"], fixed_seed=fixed_seed,
                              HEU_prob_distribution=def_HEU_mean, HEU_guilded_ratio=HEU_guilded_ratio).to(
        device)  # Global Actor Critic

    glob_AC_def.share_memory()  # share the global parameters in multiprocessing
    optim_def = SharedAdam(glob_AC_def.parameters(), lr=config_def["lr"])  # global optimizer

    on_server = False
    start_time = time.time()
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
    shared_dict['att_succ_counter_writer'] = mp.Queue()  # attack success counter
    shared_dict['att_succ_counter_ave_writer'] = mp.Queue()
    shared_dict['att_counter_writer'] = mp.Queue()  # attack launched counter
    shared_dict['att_counter_ave_writer'] = mp.Queue()
    shared_dict['att_succ_rate_writer'] = mp.Queue()  # attack success rate
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

    glob_AC_att = None
    optim_att = None
    config_att = None
    def_attribute_dict = None
    att_attribute_dict = None
    trial = None
    exp_scheme = 0
    player_name = 'def'
    is_custom_env = False
    miss_dur = None
    target_size = None
    max_att_budget = None
    num_HD = None

    num_worker = 1

    workers = [Agent(glob_AC_def, glob_AC_att, optim_def, optim_att, shared_dict=shared_dict, config_def=config_def,
                     config_att=config_att, def_attri_dict=def_attribute_dict, att_attri_dict=att_attribute_dict,
                     fixed_seed=fixed_seed, trial=trial,
                     name_id=i, exp_scheme=exp_scheme, player=player_name,
                     is_custom_env=is_custom_env, miss_dur=miss_dur,
                     target_size=target_size, max_att_budget=max_att_budget, num_HD=num_HD) for i in range(num_worker)]

    print("start workers")
    [w.start() for w in workers]

    print("join workers")
    [w.join() for w in workers]

