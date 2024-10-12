'''
@Project    ：gym-drones
@File       ：A3C_train_defender_vs_attacker.py
@Author     ：Zelin Wan
@Date       ：8/3/22
@Desciption : Given the trained defender and attacker models, and train them again under DRL-vs-DRL scenario. A2C this
time.
'''


import os
os.environ["OMP_NUM_THREADS"] = "1" # Error #34: System unable to allocate necessary resources for OMP thread:"
import time
import torch as torch
import A3C_model
import numpy as np
import random

from Gym_HoneyDrone_Defender_and_Attacker import HyperGameSim
from torch.utils.tensorboard import SummaryWriter
from A3C_model import ActorCritic
# from A3C_model import SharedAdam


def att_def_interaction(defender_model=None, attacker_model=None, is_random_scheme=True, fixed_seed=True):
    '''

    Args:
        defender_model: loaded defender model
        attacker_model: loaded attacker model
        is_random_scheme: If True, ignore loaded model and make defender and attacker randomly select strategy

    Returns:

    '''
    start_time = time.time()
    if is_random_scheme:
        scheme_name = "random_vs_random"
    else:
        scheme_name = "DRL_vs_DRL"
    # run command "tensorboard - -logdir = runs_att_def_interaction"
    writer = SummaryWriter("/Users/wanzelin/办公/gym-drones/"+scheme_name+"/runs_att_def_interaction/each_run_" +str(start_time) + "-"+scheme_name)

    config = dict(episode=100, lr=0.0001)

    print("config", config)
    print("attacker_model", attacker_model)
    print("defender_model", defender_model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using", device)

    # optimizer for attacker and defender model
    if not is_random_scheme:
        # optimizer_att = SharedAdam(attacker_model.parameters(), lr=config['lr'])
        optimizer_att = torch.optim.Adam(attacker_model.parameters(), lr=config['lr'])
        # optimizer_def = SharedAdam(defender_model.parameters(), lr=config['lr'])
        optimizer_def = torch.optim.Adam(defender_model.parameters(), lr=config['lr'])

    env = HyperGameSim(fixed_seed=fixed_seed)

    # fix seed for the whole trainning
    # env.set_random_seed()

    for episode in range(config['episode']):
        # Episode start
        step_counter = 0
        done = False
        observation = env.reset()
        observation_def = observation["def"]
        observation_att = observation["att"]
        score_def = 0
        score_att = 0
        loss_att = []
        loss_def = []
        while not done:
            if not is_random_scheme:
                action_def = defender_model.choose_action(observation_def)
                # print("action_def", action_def)
                action_att = attacker_model.choose_action(observation_att)
                # print("action_att", action_att)
                observation_new, reward, done, info = env.step(action_def=action_def, action_att=action_att)
            else:
                observation_new, reward, done, info = env.step()

            observation_new_def = observation_new["def"]
            observation_new_att = observation_new["att"]
            reward_def = reward["def"]
            reward_att = reward["att"]
            score_def += reward_def
            score_att += reward_att


            # Train Models:
            if not is_random_scheme:
                # train defender model
                # after collect 5 memory, train the model,and clear memory.
                defender_model.remember(observation_new_def, action_def, reward_def)
                if step_counter % 5 == 0 or done:
                    loss = defender_model.calc_loss(done)
                    loss_def.append(loss.item())
                    optimizer_def.zero_grad()
                    loss.backward()
                    # for local_param, global_param in zip(defender_model.parameters(), defender_model.parameters()):
                    #     print("local_param.grad", local_param.grad)
                    #     print("global_param._grad", global_param._grad)
                    #     global_param._grad = local_param.grad
                    optimizer_def.step()
                    # defender_model.load_state_dict(self.global_actor_critic.state_dict())
                    defender_model.clear_memory()

                # train attacker model
                # after collect 5 memory, train the model,and clear memory.
                attacker_model.remember(observation_new_att, action_att, reward_att)
                if step_counter % 5 == 0 or done:
                    loss = attacker_model.calc_loss(done)
                    loss_att.append(loss.item())
                    optimizer_att.zero_grad()
                    loss.backward()
                    # for local_param, global_param in zip(attacker_model.parameters(), attacker_model.parameters()):
                    #     global_param._grad = local_param.grad
                    optimizer_att.step()
                    # defender_model.load_state_dict(self.global_actor_critic.state_dict())
                    attacker_model.clear_memory()

            step_counter += 1
            observation_def = observation_new_def
            observation_att = observation_new_att

        score_def_average = score_def/step_counter
        score_att_average = score_att/ step_counter

        print('global-episode ', episode, 'score_def_average %.1f' % score_def_average, 'score_att_average %.1f' % score_att_average)
        writer.add_scalar("Defender average Score", score_def_average, episode)
        writer.add_scalar("Attacker average Score", score_att_average, episode)
        writer.add_scalar("Mission Time (step)", step_counter, episode)
        if not is_random_scheme:
            writer.add_scalar("Attacker model Loss", sum(loss_att)/len(loss_att), episode)
            writer.add_scalar("Defender model Loss", sum(loss_def) / len(loss_def), episode)

        # battery consumpiton for all drones
        consumption_all = 0
        for MD in env.system.MD_dict.values():
            consumption_all += MD.accumulated_consumption
        for HD in env.system.HD_dict.values():
            consumption_all += HD.accumulated_consumption
        writer.add_scalar("Energy Consumption", consumption_all, episode)
        if env.attacker.att_counter == 0:
            att_succ_rate = 0
        else:
            att_succ_rate = env.attacker.att_succ_counter/env.attacker.att_counter
        writer.add_scalar("Attack Success Rate", att_succ_rate, episode)
        writer.add_scalar("Mission Success Rate", env.system.scanCompletePercent(), episode)
    print("--- Simulation Time: %s seconds ---" % round(time.time() - start_time, 1))
    writer.flush()
    writer.close()  # close SummaryWriter of TensorBoard

    # save new trained models to file
    if not is_random_scheme:
        print("save models")
        model_path = "/Users/wanzelin/办公/gym-drones/train_drl_vs_drl/trained_model_drl_vs_drl/"
        os.makedirs(model_path, exist_ok=True)
        model_name_def = "trained_A3C_defender_v2"
        model_name_att = "trained_A3C_attacker_v2"
        torch.save(defender_model, model_path + model_name_def)
        torch.save(defender_model, model_path + model_name_att)
    else:
        print("The is_random_scheme = True. No model saved")


if __name__ == '__main__':
    is_random_scheme = True
    fixed_seed = True

    if not is_random_scheme:
        def_model_path = "/Users/wanzelin/办公/gym-drones/train_drl_vs_drl/trained_A3C_defender_1658376349.2343273-Trial_21"
        defender_model = torch.load(def_model_path)
        defender_model.train()
        att_model_path = "/Users/wanzelin/办公/gym-drones/train_drl_vs_drl/trained_A3C_attacker_1658385589.836977"
        attacker_model = torch.load(att_model_path)
        attacker_model.train()
        att_def_interaction(defender_model, attacker_model, is_random_scheme=is_random_scheme, fixed_seed=fixed_seed)
    else:
        att_def_interaction(is_random_scheme=is_random_scheme, fixed_seed=fixed_seed)