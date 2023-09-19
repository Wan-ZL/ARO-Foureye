'''
Project     ：Drone-DRL-HT 
File        ：utils.py
Author      ：Zelin Wan
Date        ：2/8/23
Description : Functions that use multiple times
'''

import os
# os.environ["OMP_NUM_THREADS"] = "1" # Error #34: System unable to allocate necessary resources for OMP thread:"
import torch
import numpy as np

from torch import nn
from torch.nn.utils import clip_grad_norm_


def v_wrap(np_array, dtype=np.float32):
    '''
    convert numpy array to tensor
    :param np_array:
    :param dtype:
    :return:
    '''
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)

def v_wrap_2(np_array):
    return torch.tensor(np_array, dtype=torch.float32)


def set_init(layers, fixed_seed):
    if fixed_seed:
        print("fix init seed")
        torch.manual_seed(0)
    for layer in layers:
        # nn.init.normal_(layer.weight, mean=0., std=0.1)
        nn.init.xavier_normal_(layer.weight)
        nn.init.constant_(layer.bias, 0.)

def push_and_pull_with_experience_replay(opt, lnet, gnet, done, s_, bs, ba, br, gamma):
    '''
    Update the global network using experience replay
    :param opt: optimizer for the global network
    :param lnet: local network
    :param gnet: global network
    :param done: whether the episode is done
    :param s_: new state
    :param bs: old state saved in buffer
    :param ba: action saved in buffer
    :param br: reward saved in buffer
    :param gamma: discount factor for reward
    :return: total loss, critic loss, actor loss
    '''

    # calculate v(s) (expected accumulated reward) (see 'expected accumulated reward sample check.jpg' in overleaf)
    # use equation R_{t=1} = r1 + r2 + r3 + r4 + r5 + v(s_5) to calculate the accumulated reward (add gamma when coding)
    if done:
        # terminal state
        v_s_ = 0.0
    else:
        # expected accumulated reward of the last state (V(S_5))
        v_s_ = lnet.v_net(v_wrap(s_[None, :])).data.numpy()[0, 0]

    buffer_v_target = []    # target for value network
    for r in br[::-1]:  # reverse buffer r (change order to t = 5 to 1)
        v_s_ = r + gamma * v_s_ # new v_s_ (V(S) value) will be replaced by the old one
        buffer_v_target.append(v_s_)    # save the new v_s_ to buffer_v_target
    buffer_v_target.reverse()   # reverse the order back to t = 1 to 5

    # save the temporary experience to memory buffer
    for index in range(len(bs)):
        if lnet.mem_buffer_counter < lnet.mem_buffer_size:
            # buffer not full
            lnet.mem_buffer_s[lnet.mem_buffer_counter, :] = bs[index]
            lnet.mem_buffer_a[lnet.mem_buffer_counter] = ba[index]
            lnet.mem_buffer_v_target[lnet.mem_buffer_counter] = buffer_v_target[index]
            lnet.mem_td_error[lnet.mem_buffer_counter] = 999999999999999    # initialize TD error to a large number
            lnet.mem_buffer_counter += 1
        else:
            # buffer is full
            # replace the old experience with new one
            replace_index = lnet.mem_buffer_counter % lnet.mem_buffer_size
            lnet.mem_buffer_s[replace_index, :] = bs[index]
            lnet.mem_buffer_a[replace_index] = ba[index]
            lnet.mem_buffer_v_target[replace_index] = buffer_v_target[index]
            lnet.mem_td_error[replace_index] = 999999999999999    # initialize TD error to a large number
            lnet.mem_buffer_counter += 1

    # if memory buffer is not full, do not train the network
    if lnet.mem_buffer_counter < lnet.mem_buffer_size:
        return torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

    # train N times using the memory buffer
    total_loss_set = np.array([])
    c_loss_set = np.array([])
    a_loss_set = np.array([])
    for _ in range(lnet.train_N_times):
        # selection probability for prioritized experience replay
        sample_prob = lnet.mem_td_error / lnet.mem_td_error.sum()

        # sample mini-batch from memory buffer
        sample_index = np.random.choice(lnet.mem_buffer_size, lnet.mini_batch_size, replace=False, p=sample_prob)
        bs = lnet.mem_buffer_s[sample_index, :]
        ba = lnet.mem_buffer_a[sample_index]
        buffer_v_target = lnet.mem_buffer_v_target[sample_index]

        # adjusted learning rate for prioritized experience replay
        lr_factor = (sample_prob[sample_index] * lnet.mem_buffer_size) ** (-lnet.beta)
        lnet.beta = min(lnet.beta + lnet.beta_increment_per_sampling, 1)

        # calculate loss
        total_loss, c_loss, a_loss, total_loss_weighted, td_error = lnet.loss_func(
            v_wrap(np.vstack(bs)),
            # v_wrap(np.array(ba), dtype=np.int64) if ba[0].dtype == np.int64 else v_wrap(np.vstack(ba)),
            v_wrap(np.array(ba)),
            v_wrap(np.array(buffer_v_target)), lr_factor)

        # calculate local gradients, push local parameters to global, and update global parameters
        opt.zero_grad()
        # total_loss.backward()
        total_loss_weighted.backward()
        for lp, gp in zip(lnet.parameters(), gnet.parameters()):
            gp._grad = lp.grad
        clip_grad_norm_(gnet.parameters(), gnet.max_grad_norm)
        opt.step()


        # pull global parameters
        lnet.load_state_dict(gnet.state_dict())

        # update TD error
        abs_td_error = np.abs(np.squeeze(td_error.detach().numpy()))
        eplilon = 0.00001   # avoid probability being zero
        lnet.mem_td_error[sample_index] = abs_td_error + eplilon

        # save loss
        total_loss_set = np.append(total_loss_set, total_loss.detach().numpy())
        c_loss_set = np.append(c_loss_set, c_loss.detach().numpy())
        a_loss_set = np.append(a_loss_set, a_loss.detach().numpy())


    return total_loss_set.mean(), c_loss_set.mean(), a_loss_set.mean()

def push_and_pull(opt, lnet, gnet, done, s_, bs, ba, br, gamma):
    '''
    Update the global network
    :param opt: optimizer for the global network
    :param lnet: local network
    :param gnet: global network
    :param done: whether the episode is done
    :param s_: new state
    :param bs: old state saved in buffer
    :param ba: action saved in buffer
    :param br: reward saved in buffer
    :param gamma: discount factor for reward
    :return: total loss, critic loss, actor loss
    '''

    # calculate v(s) (expected accumulated reward) (see 'expected accumulated reward sample check.jpg' in overleaf)
    # use equation R_{t=1} = r1 + r2 + r3 + r4 + r5 + v(s_5) to calculate the accumulated reward (add gamma when coding)
    if done:
        # terminal state
        v_s_ = 0.0
    else:
        # expected accumulated reward of the last state (V(S_5))
        v_s_ = lnet.v_net(v_wrap(s_[None, :])).data.numpy()[0, 0]

    buffer_v_target = []    # target for value network
    for r in br[::-1]:  # reverse buffer r (change order to t = 5 to 1)
        v_s_ = r + gamma * v_s_ # new v_s_ (V(S) value) will be replaced by the old one
        buffer_v_target.append(v_s_)    # save the new v_s_ to buffer_v_target
    buffer_v_target.reverse()   # reverse the order back to t = 1 to 5

    total_loss, c_loss, a_loss, total_loss_weighted, td_error = lnet.loss_func(
        v_wrap(np.vstack(bs)),
        v_wrap(np.array(ba), dtype=np.int64) if ba[0].dtype == np.int64 else v_wrap(np.vstack(ba)),
        v_wrap(np.array(buffer_v_target)[:, None]))


    # calculate local gradients and push local parameters to global
    opt.zero_grad()
    total_loss.backward()
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        gp._grad = lp.grad
    opt.step()

    # pull global parameters
    lnet.load_state_dict(gnet.state_dict())
    return total_loss.detach().numpy(), c_loss.detach().numpy(), a_loss.detach().numpy()


def record(global_ep, global_ep_r, ep_r, res_queue, name):
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
    res_queue.put(global_ep_r.value)
    print(
        name,
        "Ep:", global_ep.value,
        "| Ep_r: %.0f" % global_ep_r.value,
    )


def get_last_ten_ave(list):
    len_last_return = max(1, int(len(list) * 0.1))  # max can make sure at lead one element in list
    last_ten_percent_return = list[-len_last_return:]
    if len(last_ten_percent_return):
        ave_10_per_return = sum(last_ten_percent_return) / len(last_ten_percent_return)
    else:
        ave_10_per_return = 0

    return ave_10_per_return