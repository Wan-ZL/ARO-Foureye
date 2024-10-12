import os
os.environ["OMP_NUM_THREADS"] = "1" # Error #34: System unable to allocate necessary resources for OMP thread:"

import ctypes

import pickle
import time
from copy import copy
import gym
import torch as torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from Gym_HoneyDrone_Defender_and_Attacker import HyperGameSim
from multiprocessing import Manager
from torch.utils.tensorboard import SummaryWriter
from A3C_model import *


def att_def_interaction(defender_model, attacker_model):
    start_time = time.time()
    # run command "tensorboard --logdir=runs_att_def_interaction"
    scenario = 'random'    # 'random', or 'att, or 'def', or 'att_def'
    writer = SummaryWriter("runs_att_def_interaction/each_run_" +str(start_time) + '_' + scenario)

    config = dict(episode=100)    # this config may be changed by optuna

    print("config", config)
    print("attacker_model", attacker_model)
    print("defender_model", defender_model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using", device)

    env = HyperGameSim()

    for episode in range(config['episode']):
        # Episode start
        step_counter = 0
        done = False
        observation = env.reset()
        obser_def = observation['def']
        obser_att = observation['att']
        score_def = 0
        score_att = 0
        att_succ_counter_one_epi = 0
        att_counter_one_epi = 0
        while not done:
            # observation, reward_def, reward_att, done, info = env.step(action_def=action_def, action_att=action_att)
            if scenario == 'att':
                action_att = attacker_model.choose_action(obser_att)
                observation, reward, done, info = env.step(action_att=action_att)
            elif scenario == 'def':
                action_def = defender_model.choose_action(obser_def)
                observation, reward, done, info = env.step(action_def=action_def)
            elif scenario == 'att_def':
                action_att = attacker_model.choose_action(obser_att)
                action_def = defender_model.choose_action(obser_def)
                observation, reward, done, info = env.step(action_def=action_def, action_att=action_att)
            elif scenario == 'random':
                observation, reward, done, info = env.step()
            else:
                raise Exception("invalide scenario")
            # unpack observation and reward for attacker and defender
            obser_def = observation['def']
            obser_att = observation['att']
            reward_def = reward['def']
            reward_att = reward['att']
            score_def += reward_def
            score_att += reward_att
            att_succ_counter_one_epi += env.attacker.att_succ_counter
            att_counter_one_epi += env.attacker.att_counter
            step_counter += 1
        score_def_average = score_def/step_counter
        score_att_average = score_att/ step_counter
        print('global-episode ', episode, 'score_def_average %.1f' % score_def_average, 'score_att_average %.1f' % score_att_average)
        writer.add_scalar("Defender average Score", score_def_average, episode)
        writer.add_scalar("Attacker average Score", score_att_average, episode)
        writer.add_scalar("Mission Time (step)", step_counter, episode)
        # battery consumpiton for all drones
        consumption_all = 0
        for MD in env.system.MD_dict.values():
            consumption_all += MD.accumulated_consumption
        for HD in env.system.HD_dict.values():
            consumption_all += HD.accumulated_consumption
        writer.add_scalar("Energy Consumption", consumption_all, episode)
        if att_counter_one_epi == 0:
            att_succ_rate = 0
        else:
            att_succ_rate = att_succ_counter_one_epi/att_counter_one_epi
        writer.add_scalar("Attack Success Rate", att_succ_rate, episode)
        writer.add_scalar("Mission Completion Progress", env.system.scanCompletePercent(), episode)
        writer.add_scalar("Mission Result (1 succ, 0 fail)", 1 if env.system.scanCompletePercent() == 1 else 0, episode)
    print("--- Simulation Time: %s seconds ---" % round(time.time() - start_time, 1))
    writer.flush()
    writer.close()  # close SummaryWriter of TensorBoard


if __name__ == '__main__':
    # def_model_path = "/Users/wanzelin/办公/gym-drones/data_for_defender/model/trained_A3C_defender_1658376349.2343273-Trial_21"
    def_model_path = "/Users/wanzelin/办公/gym-drones/train_drl_vs_drl/trained_model_drl_vs_drl/trained_A3C_defender_v2"
    defender_model = torch.load(def_model_path)
    defender_model.eval()
    # att_model_path = "/Users/wanzelin/办公/gym-drones/data_for_attacker/model/trained_A3C_attacker_1658385589.836977"
    att_model_path = "/Users/wanzelin/办公/gym-drones/train_drl_vs_drl/trained_model_drl_vs_drl/trained_A3C_attacker_v2"
    attacker_model = torch.load(att_model_path)
    attacker_model.eval()
    att_def_interaction(defender_model, attacker_model)


