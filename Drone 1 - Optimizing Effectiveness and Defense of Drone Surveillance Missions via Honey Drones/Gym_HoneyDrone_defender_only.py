from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
import gym
import math
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from model_System import system_model
from model_Defender import defender_model
from model_Attacker import attacker_model
from multiprocessing import cpu_count
from multiprocessing import Process

import time
import argparse
import pybullet as p
import gym

import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from model_System import system_model
from model_Defender import defender_model
from model_Attacker import attacker_model
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.utils import sync, str2bool


class HyperGameSim(Env):
    def __init__(self):
        # variable for bullet drone env
        self.print = False
        self.gui = False
        self.system = None
        self.defender = None
        self.attacker = None
        self.frameN = 1
        self.update_freq = 500
        self.TARGET_POS = None
        self.ctrl = None
        self.CTRL_EVERY_N_STEPS = None
        self.env = None
        self.pybullet_action = None
        self.START = None
        self.ARGS = None
        self.start_time = None
        self.initDroneEnv()
        self.close_env()  # close client in init for avoiding client limit error

        # variable for current env
        self.action_space = Discrete(self.defender.strategy)
        # [time duration, ratio of mission complete]
        self.observation_space = Box(low=np.array([0., 0.]),
                                     high=np.array([self.defender.system.mission_max_duration, 1.]))

    # close client. This function check the connection status before close it. Use this function to avoid client limit.
    def close_env(self):
        if self.env is not None:
            if p.getConnectionInfo(self.env.getPyBulletClient())['isConnected']:
                # print("closing client", self.env.getPyBulletClient())
                self.env.close()

    def reset(self, *args):
        self.start_time = time.time()
        self.initDroneEnv()
        # mission_complete_ratio = self.system.scanCompletePercent()
        state = [self.system.mission_duration, self.system.scanCompletePercent()]
        return state

    def initDroneEnv(self):
        # create model class
        self.system = system_model()
        self.defender = defender_model(self.system)
        self.attacker = attacker_model(self.system)

        if self.print: print("attacker locaiton", self.attacker.xyz)


        #### default parameters:
        num_MD = self.system.num_MD  # number of MD (in index, MD first then HD)
        num_HD = self.system.num_HD  # number of HD

        #### Define and parse (optional) arguments for the script ##
        parser = argparse.ArgumentParser(
            description='Helix flight script using CtrlAviary or VisionAviary and DSLPIDControl')
        parser.add_argument('--drone', default="cf2p", type=DroneModel, help='Drone model (default: CF2X)', metavar='',
                            choices=DroneModel)
        parser.add_argument('--num_drones', default=num_HD + num_MD, type=int, help='Number of drones', metavar='')
        parser.add_argument('--physics', default="pyb", type=Physics, help='Physics updates (default: PYB)', metavar='',
                            choices=Physics)
        parser.add_argument('--vision', default=False, type=str2bool,
                            help='Whether to use VisionAviary (default: False)', metavar='')
        parser.add_argument('--gui', default=self.gui, type=str2bool,
                            help='Whether to use PyBullet GUI (default: True)',
                            metavar='')
        parser.add_argument('--record_video', default=False, type=str2bool,
                            help='Whether to record a video (default: False)', metavar='')
        parser.add_argument('--plot', default=True, type=str2bool,
                            help='Whether to plot the simulation results (default: True)', metavar='')
        parser.add_argument('--user_debug_gui', default=False, type=str2bool,
                            help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
        parser.add_argument('--aggregate', default=True, type=str2bool,
                            help='Whether to aggregate physics steps (default: True)', metavar='')
        parser.add_argument('--obstacles', default=False, type=str2bool,
                            help='Whether to add obstacles to the environment (default: False)', metavar='')
        parser.add_argument('--simulation_freq_hz', default=30, type=int,
                            help='Simulation frequency in Hz (default: 240)', metavar='')
        parser.add_argument('--control_freq_hz', default=20, type=int, help='Control frequency in Hz (default: 48)',
                            metavar='')
        parser.add_argument('--duration_sec', default=5, type=int,
                            help='Duration of the simulation in seconds (default: 5)', metavar='')
        self.ARGS = parser.parse_args()

        #### Initialize the simulation #############################
        H = 1
        H_STEP = .05
        R = .3

        INIT_XYZS = np.array([[R * np.cos((i / 6) * 2 * np.pi + np.pi / 2),
                               R * np.sin((i / 6) * 2 * np.pi + np.pi / 2) - R, H + i * H_STEP] for i in
                              range(self.ARGS.num_drones)])

        INIT_RPYS = np.array([[0, 0, i * (np.pi / 2) / self.ARGS.num_drones] for i in range(self.ARGS.num_drones)])
        AGGR_PHY_STEPS = int(self.ARGS.simulation_freq_hz / self.ARGS.control_freq_hz) if self.ARGS.aggregate else 1

        # gym-like environment
        self.close_env()  # close previous client
        self.env = CtrlAviary(drone_model=self.ARGS.drone,
                              num_drones=self.ARGS.num_drones,
                              initial_xyzs=INIT_XYZS,
                              initial_rpys=INIT_RPYS,
                              physics=self.ARGS.physics,
                              neighbourhood_radius=10,
                              freq=self.ARGS.simulation_freq_hz,
                              aggregate_phy_steps=AGGR_PHY_STEPS,
                              gui=self.ARGS.gui,
                              record=self.ARGS.record_video,
                              obstacles=self.ARGS.obstacles,
                              user_debug_gui=self.ARGS.user_debug_gui,
                              print_result = False
                              )
        # print("creating client", self.env.getPyBulletClient())

        PYB_CLIENT = self.env.getPyBulletClient()

        #### Initialize the controllers ############################
        self.ctrl = [DSLPIDControl(drone_model=self.ARGS.drone) for i in range(self.ARGS.num_drones)]

        #### Run the simulation ####################################
        #### (0,0) is base station ####
        self.CTRL_EVERY_N_STEPS = int(np.floor(self.env.SIM_FREQ / self.ARGS.control_freq_hz))
        if self.print: print("CTRL_EVERY_N_STEPS", self.CTRL_EVERY_N_STEPS)
        self.pybullet_action = {str(i): np.array([12713, 12713, 12713, 12713]) for i in range(self.ARGS.num_drones)}
        self.START = time.time()
        self.frameN = 1

        # initial position for drones (MD+HD)
        for MD in self.system.MD_set:
            MD.assign_destination(INIT_XYZS[MD.ID])
            MD.xyz_temp = MD.xyz
        for HD in self.system.HD_set:
            HD.assign_destination(INIT_XYZS[HD.ID])
            HD.xyz_temp = HD.xyz

        # disconnect exist physics client, and create a new clident
        p.loadURDF("duck_vhacd.urdf", self.attacker.xyz, physicsClientId=PYB_CLIENT)

        self.update_freq = self.system.update_freq




    def step(self, action_def):

        # pybullet environment
        self.roundBegin(action_def)
        self.moveDrones()

        # hyperGame environment
        mission_complete_ratio = self.system.scanCompletePercent()
        state = [self.system.mission_duration, mission_complete_ratio]
        w_MS = 0.4
        w_AC = 0.3
        w_MT = 0.3

        if mission_complete_ratio:
            reward = w_MS * mission_complete_ratio + w_AC * self.system.aliveDroneCount() / (
                        self.system.num_MD + self.system.num_HD) + w_MT * self.system.mission_duration / self.system.mission_duration_max
        else:
            reward = 0
        if self.system.is_mission_Not_end():
            done = False
        else:
            done = True
            self.close_env()  # close client for avoiding client limit error
            print("--- game duration: %s seconds ---" % round(time.time() - self.start_time, 1))


        info = {}
        info["mission condition"] = self.system.mission_condition
        return state, reward, done, info

    def roundBegin(self, action_def):
        # check if mission complete
        self.system.check_mission_complete()

        # path planning for MD
        if self.defender.is_calc_trajectory():
            MD_trajectory = self.defender.MD_trajectory

        # Drones state update
        self.system.Drone_state_update()

        # show real time scan
        if self.print: print("map \n", self.system.scan_cell_map)

        # for MD update next destination
        if self.frameN <= self.update_freq:  # first round doesn't check scan_map
            self.defender.update_MD_next_destination_no_scanMap()
        else:
            self.defender.update_MD_next_destination()

        # avoid HD crash when creating
        if self.frameN <= self.update_freq:
            for HD in self.system.HD_set:
                HD.assign_destination(self.defender.HD_locations[HD.ID])

        # # for HD update next destination
        self.defender.update_HD_next_destination()

        # Attacker
        self.attacker.observe()
        self.attacker.select_strategy(random.randint(1, 9))
        self.attacker.action()

        # Defender
        self.defender.select_strategy(action_def)
        self.defender.action()

    def moveDrones(self):
        for _ in range(self.update_freq):
            self.updateTempLoca(self.system.MD_set, self.ARGS.control_freq_hz)
            self.updateTempLoca(self.system.HD_set, self.ARGS.control_freq_hz)

            obs, reward, done, info = self.env.step(self.pybullet_action)

            # scan count, and check if crashed
            self.system.MD_environment_interaction(obs)
            self.system.HD_environment_interaction(obs)

            # energy consumption of MD and HD
            self.system.battery_consume()

            # execute pybullet_action
            for MD in self.system.MD_set:
                if MD.crashed:
                    continue
                self.pybullet_action[str(MD.ID)], _, _ = self.ctrl[MD.ID].computeControlFromState(
                    control_timestep=self.CTRL_EVERY_N_STEPS * self.env.TIMESTEP,
                    state=obs[str(MD.ID)]["state"],
                    target_pos=MD.xyz_temp)
            for HD in self.system.HD_set:
                self.pybullet_action[str(HD.ID)], _, _ = self.ctrl[HD.ID].computeControlFromState(
                    control_timestep=self.CTRL_EVERY_N_STEPS * self.env.TIMESTEP,
                    state=obs[str(HD.ID)]["state"],
                    target_pos=HD.xyz_temp)

            self.frameN += 1

            #### Sync the simulation ###################################
            if self.ARGS.gui:
                sync(self.frameN, self.START, self.env.TIMESTEP)

    def render(self, *args):
        pass



    def updateTempLoca(self, drone_set, control_freq_hz: int):
        for drone in drone_set:
            if drone.crashed:
                continue
            vector_per_frame = (drone.xyz - drone.xyz_temp) / (
                    2 * control_freq_hz)  # multiple 2 to slow down drone speed for a long distance travel

            # avoid too fast movement. It cause crash
            vector_magnitude = np.linalg.norm(vector_per_frame)
            if vector_magnitude > drone.speed_per_frame_max:
                vector_per_frame = vector_per_frame / (vector_magnitude / drone.speed_per_frame_max)
            drone.xyz_temp = vector_per_frame + drone.xyz_temp

    def drones_distance(self, drone_x_location, drone_y_location):
        distance_squre = np.square(drone_x_location - drone_y_location)
        return np.sqrt(np.sum(distance_squre))
