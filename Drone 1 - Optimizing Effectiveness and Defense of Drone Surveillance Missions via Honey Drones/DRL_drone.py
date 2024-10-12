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

def updateTempLoca(drone_set, control_freq_hz: int):
    for drone in drone_set:
        if drone.crashed:
            continue
        vector_per_frame = (drone.xyz - drone.xyz_temp) / (2 * control_freq_hz)  # multiple 2 to slow down drone speed for a long distance travel

        # avoid too fast movement. It cause crash
        vector_magnitude = np.linalg.norm(vector_per_frame)
        if vector_magnitude > drone.speed_per_frame_max:
            vector_per_frame = vector_per_frame / (vector_magnitude/drone.speed_per_frame_max)
        drone.xyz_temp = vector_per_frame + drone.xyz_temp

def drones_distance(drone_x_location, drone_y_location):
    distance_squre = np.square(drone_x_location - drone_y_location)
    return np.sqrt(np.sum(distance_squre))


if __name__ == "__main__":
    # create model class
    system = system_model()
    defender = defender_model(system)
    attacker = attacker_model(system)
    MD_dict = system.MD_dict
    HD_dict = system.HD_dict
    print("attacker locaiton", attacker.xyz)

    # sample for obtain general parameter of MD and HD
    HD_sample = system.sample_HD
    MD_sample = system.sample_MD

    #### default parameters:
    num_MD = system.num_MD # number of MD (in index, MD first then HD)
    num_HD = system.num_HD  # number of HD
    sg_HD = defender.strategy # defense strategy, range [0, 10]
    tao_lower = defender.tau_lower   # The lower bounds of the number of MDs that HDs can protect simultaneously
    tao_upper = defender.tau_upper   # The upper bounds of the number of MDs that HDs can protect simultaneously
    map_size = system.map_size # size of surveillance area (map size)

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary or VisionAviary and DSLPIDControl')
    parser.add_argument('--drone',              default="cf2p",     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=num_HD + num_MD, type=int, help='Number of drones', metavar='')
    parser.add_argument('--physics',            default="pyb",      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--vision',             default=False,      type=str2bool,      help='Whether to use VisionAviary (default: False)', metavar='')
    parser.add_argument('--gui',                default=True,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=False,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=True,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=False,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--aggregate',          default=True,       type=str2bool,      help='Whether to aggregate physics steps (default: True)', metavar='')
    parser.add_argument('--obstacles',          default=False,      type=str2bool,      help='Whether to add obstacles to the environment (default: False)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=240,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=48,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=12,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    ARGS = parser.parse_args()

    #### Initialize the simulation #############################
    H = 1
    H_STEP = .05
    R = .3

    INIT_XYZS = np.array([[R*np.cos((i/6)*2*np.pi+np.pi/2), R*np.sin((i/6)*2*np.pi+np.pi/2)-R, H+i*H_STEP] for i in range(ARGS.num_drones)])

    INIT_RPYS = np.array([[0, 0,  i * (np.pi/2)/ARGS.num_drones] for i in range(ARGS.num_drones)])
    AGGR_PHY_STEPS = int(ARGS.simulation_freq_hz/ARGS.control_freq_hz) if ARGS.aggregate else 1

    # gym-like environment
    env = CtrlAviary(drone_model=ARGS.drone,
                     num_drones=ARGS.num_drones,
                     initial_xyzs=INIT_XYZS,
                     initial_rpys=INIT_RPYS,
                     physics=ARGS.physics,
                     neighbourhood_radius=10,
                     freq=ARGS.simulation_freq_hz,
                     aggregate_phy_steps=AGGR_PHY_STEPS,
                     gui=ARGS.gui,
                     record=ARGS.record_video,
                     obstacles=ARGS.obstacles,
                     user_debug_gui=ARGS.user_debug_gui
                     )


    PYB_CLIENT = env.getPyBulletClient()
    PERIOD = 10
    NUM_WP = ARGS.control_freq_hz*PERIOD
    TARGET_POS = np.zeros((NUM_WP,3))
    wp_counters = np.array([int((i*NUM_WP/6)%NUM_WP) for i in range(ARGS.num_drones)])

    #### Initialize the controllers ############################
    ctrl = [DSLPIDControl(drone_model=ARGS.drone) for i in range(ARGS.num_drones)]

    #### Run the simulation ####################################
    #### (0,0) is base station ####
    CTRL_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ/ARGS.control_freq_hz))
    print("CTRL_EVERY_N_STEPS", CTRL_EVERY_N_STEPS)
    action = {str(i): np.array([12713,12713,12713,12713, 20]) for i in range(ARGS.num_drones)}
    START = time.time()
    frameN = 1

    # initial target scan area/map
    map_x = system.map_ori_x   # original point of target area
    map_y = system.map_ori_y

    map_border = 0  # create a boarder to avoid 'index error'
    map_x_index_const = map_x - map_border
    map_y_index_const = map_y - map_border
    map_size_with_border = map_size + 1 + (2 * map_border)
    print("map_size_with_border", map_size_with_border)
    scan_map = np.zeros((map_size_with_border,map_size_with_border))

    # initial position for drones (MD+HD)
    for MD in system.MD_set:
        MD.assign_destination(INIT_XYZS[MD.ID])
        MD.xyz_temp = MD.xyz
    for HD in system.HD_set:
        HD.assign_destination(INIT_XYZS[HD.ID])
        HD.xyz_temp = HD.xyz

    MD_trajectory = defender.MD_trajectory
    p.loadURDF("duck_vhacd.urdf", attacker.xyz, physicsClientId=PYB_CLIENT)

    # ==== PyTorch: Initialization ====
    use_PyTorch = False
    if use_PyTorch:
        # set up matplotlib
        is_ipython = 'inline' in matplotlib.get_backend()
        if is_ipython:
            from IPython import display
        plt.ion()

        # if gpu is to be used
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ==== Pytorch: Replay Memory ====
        Transition = namedtuple('Transition', ('state', 'pybullet_action', 'next_state', 'reward'))

        class ReplayMemory(object):
            def __init__(self, capacity):
                self.memory = deque([], maxlen=capacity)

            def push(self, *args):
                """Save a transition"""
                self.memory.append(Transition(*args))

            def sample(self, batch_size):
                return random.sample(self.memory, batch_size)

            def __len__(self):
                return len(self.memory)

        # ==== Pytorch: Q-network ====
        class DQN(nn.Module):
            def __init__(self, state_space_dim, action_space_dim):
                super(DQN, self).__init__()
                self.linear = nn.Sequential(
                    nn.Linear(state_space_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 64*2),
                    nn.ReLU(),
                    nn.Linear(64*2, action_space_dim)
                )

            def forward(self, x):
                x = x.to(device)
                return self.linear(x)

            # def __init__(self, h, w, outputs):
            #     super(DQN, self).__init__()
            #     self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
            #     self.bn1 = nn.BatchNorm2d(16)
            #     self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
            #     self.bn2 = nn.BatchNorm2d(32)
            #     self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
            #     self.bn3 = nn.BatchNorm2d(32)
            #
            #     # Number of Linear input connections depends on output of conv2d layers
            #     # and therefore the input image size, so compute it.
            #     def conv2d_size_out(size, kernel_size=5, stride=2):
            #         return (size - (kernel_size - 1) - 1) // stride + 1
            #
            #     convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
            #     convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
            #     linear_input_size = convw * convh * 32
            #     self.head = nn.Linear(linear_input_size, outputs)
            #
            # # Called with either one element to determine next pybullet_action, or a batch
            # # during optimization. Returns tensor([[left0exp,right0exp]...]).
            # def forward(self, x):
            #     x = x.to(device)
            #     x = F.relu(self.bn1(self.conv1(x)))
            #     x = F.relu(self.bn2(self.conv2(x)))
            #     x = F.relu(self.bn3(self.conv3(x)))
            #     return self.head(x.view(x.size(0), -1))


        # ==== PyTorch: Define exploration profile ====
        initial_value = 5
        num_iterations = 800
        exp_decay = np.exp(-np.log(initial_value) / num_iterations * 6)
        exploration_profile = [initial_value * (exp_decay ** i) for i in range(num_iterations)]

        plt.figure(figsize=(12, 8))
        plt.plot(exploration_profile)
        plt.grid()
        plt.xlabel('Iteration')
        plt.ylabel('Exploration profile (Softmax temperature)')

        # ==== PyTorch: Training ====
        env.seed(0)


        state_space_dim = env.observation_space.shape[0]
        action_space_dim = env.action_space.n
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")






    # initial env
    obs, reward, done, info = env.step(action)


    update_freq = system.update_freq
    while system.is_mission_Not_end():
        # update destination for drones (every 500 frames)
        if frameN % update_freq == 0:

            # check if mission complete
            system.check_mission_complete()

            # path planning for MD
            if defender.is_calc_trajectory():
                MD_trajectory = defender.MD_trajectory

            # Drones state update
            system.Drone_state_update()

            # show real time scan
            # print_debug("map \n", scan_map.round(1))
            print("map \n", system.scan_cell_map)

            # for MD update next destination
            if frameN <= update_freq:   # first round doesn't check scan_map
                defender.update_MD_next_destination_no_scanMap()
            else:
                defender.update_MD_next_destination()

            # avoid HD crash when creating
            if frameN <= update_freq:
                for HD in system.HD_set:
                    HD.assign_destination(defender.HD_locations[HD.ID])

            # # for HD update next destination
            defender.update_HD_next_destination()

            # Attacker
            attacker.observe()
            attacker.select_strategy()
            attacker.action()

            # Defender

            defender.select_strategy()
            defender.action()

        updateTempLoca(system.MD_set, ARGS.control_freq_hz)
        updateTempLoca(system.HD_set, ARGS.control_freq_hz)


        H_STEP = .05
        R = .3
        Tar_H = np.random.uniform(0,1,1)

        obs, reward, done, info = env.step(action)

        # scan count, and check if crashed
        system.MD_environment_interaction(obs)
        system.HD_environment_interaction(obs)

        # energy consumption of MD and HD
        system.battery_consume()

        # execute pybullet_action
        for MD in system.MD_set:
            if MD.crashed:
                continue
            action[str(MD.ID)], _, _ = ctrl[MD.ID].computeControlFromState(control_timestep=CTRL_EVERY_N_STEPS * env.TIMESTEP,
                                                                   state=obs[str(MD.ID)]["state"],
                                                                   target_pos=MD.xyz_temp)
        for HD in system.HD_set:
            action[str(HD.ID)], _, _ = ctrl[HD.ID].computeControlFromState(control_timestep=CTRL_EVERY_N_STEPS * env.TIMESTEP,
                                                                   state=obs[str(HD.ID)]["state"],
                                                                   target_pos=HD.xyz_temp)

        frameN += 1

        #### Sync the simulation ###################################
        if ARGS.gui:
            sync(frameN, START, env.TIMESTEP)










