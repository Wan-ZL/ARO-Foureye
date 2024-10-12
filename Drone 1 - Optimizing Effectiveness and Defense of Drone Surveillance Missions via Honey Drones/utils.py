"""
Functions that use multiple times
"""

import os
# os.environ["OMP_NUM_THREADS"] = "1" # Error #34: System unable to allocate necessary resources for OMP thread:"
import torch
import numpy as np

from torch import nn


def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)


def set_init(layers, fixed_seed):
    if fixed_seed:
        print("fix init seed")
        torch.manual_seed(0)
    for layer in layers:
        # nn.init.normal_(layer.weight, mean=0., std=0.1)
        nn.init.xavier_normal_(layer.weight)
        nn.init.constant_(layer.bias, 0.)


def push_and_pull(opt, lnet, gnet, done, s_, bs, ba, br, gamma):
    if done:
        v_s_ = 0.               # terminal
    else:
        # v_s_ = lnet.forward(v_wrap(s_[None, :]))[-1].data.numpy()[0, 0]
        v_s_ = lnet.v_net(v_wrap(s_[None, :])).data.numpy()[0, 0]


    buffer_v_target = []
    for r in br[::-1]:    # reverse buffer r
        v_s_ = r + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()

    total_loss, c_loss, a_loss = lnet.loss_func(
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
    return total_loss, c_loss, a_loss


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