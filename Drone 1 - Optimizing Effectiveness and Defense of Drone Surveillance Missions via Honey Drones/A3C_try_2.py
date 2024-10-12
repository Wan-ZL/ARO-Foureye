'''
Project     ：gym-drones 
File        ：A3C_try_2.py
Author      ：Zelin Wan
Date        ：9/1/22
Description : 
'''

import gym
import torch as T
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from utils import push_and_pull


class SharedAdam(T.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps,
                                         weight_decay=weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = T.zeros_like(p.data)
                state['exp_avg_sq'] = T.zeros_like(p.data)

                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions, gamma=0.99):
        super(ActorCritic, self).__init__()

        self.gamma = gamma
        T.manual_seed(1)
        self.pi1 = nn.Linear(*input_dims, 128)
        self.v1 = nn.Linear(*input_dims, 128)
        self.pi = nn.Linear(128, n_actions)
        self.v = nn.Linear(128, 1)
        self.init_weight_bias(self.pi1)
        self.init_weight_bias(self.v1)
        self.init_weight_bias(self.pi)
        self.init_weight_bias(self.v)

        self.rewards = []
        self.actions = []
        self.states = []

    def init_weight_bias(self, layer):
        if type(layer) == nn.Linear:
            nn.init.xavier_normal_(layer.weight)  # use normal distribution
            # nn.init.normal_(layer.bias, std=1 / layer.in_features)
            # nn.init.normal_(layer.weight, mean=0., std=0.1)
            nn.init.constant_(layer.bias, 0.)


    def remember(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def forward(self, state):
        pi1 = F.relu(self.pi1(state))
        v1 = F.relu(self.v1(state))

        pi = self.pi(pi1)
        v = self.v(v1)

        return pi, v

    # def calc_R(self, done, observation_new):
    #     states = T.tensor(self.states, dtype=T.float)
    #     observation_new = T.tensor(observation_new, dtype=T.float)
    #     # _, v = self.forward(states)
    #     # R = v[-1] * (1 - int(done))
    #
    #     _, v_next = self.forward(observation_new)
    #     R = v_next * (1 - int(done))
    #
    #
    #     batch_return = []
    #     for reward in self.rewards[::-1]:
    #         R = reward + self.gamma * R
    #         batch_return.append(R)
    #     batch_return.reverse()
    #     batch_return = T.tensor(batch_return, dtype=T.float)
    #
    #     return batch_return

    def calc_R(self, done):
        states = T.tensor(self.states, dtype=T.float)
        _, v = self.forward(states)

        R = v[-1] * (1 - int(done))

        batch_return = []
        for reward in self.rewards[::-1]:
            R = reward + self.gamma * R
            batch_return.append(R)
        batch_return.reverse()
        batch_return = T.tensor(batch_return, dtype=T.float)

        return batch_return

    # def calc_R(self, done, observation_new):
    #     states = T.tensor(self.states, dtype=T.float)
    #     observation_new = T.tensor(observation_new, dtype=T.float)
    #     # _, v = self.forward(states)
    #     # v = self.v_net(states)
    #     v_next = self.v_net(observation_new)    # v for the state_{t_max+1}
    #
    #     # TODO: discuss, should I use v_net(states[-1] or v_net(observation_new) ?
    #     R = v_next * (1 - int(done))
    #
    #     batch_return = []
    #     for reward in self.rewards[::-1]:
    #         R = reward + self.gamma * R
    #         batch_return.append(R)
    #     batch_return.reverse()
    #     batch_return = T.tensor(batch_return, dtype=T.float)
    #
    #     return batch_return

    # def calc_loss(self, done, observation_new):
    #     states = T.tensor(self.states, dtype=T.float)
    #     actions = T.tensor(self.actions, dtype=T.float)
    #
    #     returns = self.calc_R(done, observation_new)
    #
    #     pi, values = self.forward(states)
    #     values = values.squeeze()
    #     critic_loss = (returns - values) ** 2
    #
    #     probs = T.softmax(pi, dim=1)
    #     dist = Categorical(probs)   # normalize
    #     log_probs = dist.log_prob(actions)  # log(prob)
    #     actor_loss = -log_probs * (returns - values)
    #     total_loss = (critic_loss + actor_loss).mean()
    #     print("critic_loss", critic_loss, "actor_loss", actor_loss, "total_loss", total_loss)
    #     return total_loss

    # def calc_loss(self, done):
    def loss_func(self, states, actions, returns):
        self.train()
        # states = T.tensor(self.states, dtype=T.float)
        # actions = T.tensor(self.actions, dtype=T.float)

        # returns = self.calc_R(done)
        # print("returns", returns)
        logits, values = self.forward(states)
        td = returns - values
        c_loss = td.pow(2)

        probs = F.softmax(logits, dim=1)
        m = Categorical(probs)
        exp_v = m.log_prob(actions) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        # print(c_loss,  a_loss, total_loss)
        return total_loss


    # def calc_loss(self, done, observation_new):
    #     states = T.tensor(self.states, dtype=T.float)
    #     actions = T.tensor(self.actions, dtype=T.float)
    #
    #     returns = self.calc_R(done, observation_new)     # TODO: find out what is this
    #
    #     pi = self.pi_net(states)
    #     probs = T.softmax(pi, dim=1)
    #
    #     values = self.v_net(states)
    #
    #     values = values.squeeze()
    #     critic_loss = (returns - values) ** 2
    #
    #     dist = Categorical(probs)
    #     log_probs = dist.log_prob(actions)
    #     actor_loss = -log_probs * (returns - values)
    #
    #     total_loss = (critic_loss + actor_loss).mean()
    #     # print("total_loss", total_loss)
    #     # print("critic_loss", critic_loss)
    #     # print("actor_loss", actor_loss)
    #     return total_loss, critic_loss.mean(), actor_loss.mean()

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float)
        pi, v = self.forward(state)
        probs = T.softmax(pi, dim=1)
        dist = Categorical(probs)
        action = dist.sample().numpy()[0]

        return action


class Agent(mp.Process):
    def __init__(self, global_actor_critic, optimizer, input_dims, n_actions,
                 gamma, lr, name, global_ep_idx, env_id):
        super(Agent, self).__init__()
        self.local_actor_critic = ActorCritic(input_dims, n_actions, gamma)
        self.global_actor_critic = global_actor_critic
        self.name = 'w%02i' % name
        self.episode_idx = global_ep_idx
        self.env = gym.make(env_id)
        self.optimizer = optimizer
        self.gamma = gamma

    def run(self):
        t_step = 1
        N_GAMES = 10000
        T_MAX = 5
        while self.episode_idx.value < N_GAMES:
            done = False
            observation = self.env.reset()
            score = 0
            self.local_actor_critic.clear_memory()
            while not done:
                # self.env.render(mode = "human")
                action = self.local_actor_critic.choose_action(observation)
                observation_, reward, done, info = self.env.step(action)
                score += reward
                self.local_actor_critic.remember(observation, action, reward)

                if t_step % T_MAX == 0 or done:
                    push_and_pull(self.optimizer, self.local_actor_critic, self.global_actor_critic, done, observation_, self.local_actor_critic.states, self.local_actor_critic.actions, self.local_actor_critic.rewards, self.gamma)

                    # loss = self.local_actor_critic.calc_loss(done)
                    # self.optimizer.zero_grad()
                    # loss.backward()
                    # for local_param, global_param in zip(self.local_actor_critic.parameters(),
                    #                                      self.global_actor_critic.parameters()):
                    #     global_param._grad = local_param.grad
                    # self.optimizer.step()
                    # self.local_actor_critic.load_state_dict(
                    #     self.global_actor_critic.state_dict())

                    self.local_actor_critic.clear_memory()
                t_step += 1
                observation = observation_

            with self.episode_idx.get_lock():
                self.episode_idx.value += 1
            # if score < 11:
            #     print("self.global_actor_critic.state_dict()", self.global_actor_critic.state_dict())
            print(self.name, 'episode ', self.episode_idx.value, 'reward %.1f' % score)


if __name__ == '__main__':
    lr = 1e-4
    env_id = 'CartPole-v1'
    n_actions = 2
    input_dims = [4]
    N_GAMES = 100000
    T_MAX = 5
    global_actor_critic = ActorCritic(input_dims, n_actions)
    global_actor_critic.share_memory()
    optim = SharedAdam(global_actor_critic.parameters(), lr=lr,
                       betas=(0.92, 0.999))
    global_ep = mp.Value('i', 0)

    worker_num = mp.cpu_count()
    # worker_num = 4
    workers = [Agent(global_actor_critic,
                     optim,
                     input_dims,
                     n_actions,
                     gamma=0.99,
                     lr=lr,
                     name=i,
                     global_ep_idx=global_ep,
                     env_id=env_id) for i in range(worker_num)]
    [w.start() for w in workers]
    [w.join() for w in workers]