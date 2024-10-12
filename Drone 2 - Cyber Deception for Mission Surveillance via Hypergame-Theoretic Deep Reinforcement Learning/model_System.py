'''
Project     ：Drone-DRL-HT 
File        ：model_System.py
Author      ：Zelin Wan
Date        ：2/4/23
Description : 
'''


import numpy as np
from model_MD import Mission_Drone
from model_HD import Honey_Drone
from model_RLD import RLD_Drone
import networkx as nx
import matplotlib.pyplot as plt


class system_model:
    def __init__(self, mission_duration_max=150, map_cell_number=5, num_HD=5):
        self.print = False
        self.mission_Not_end = 2  # 2 means True, use 2 here to allow drone back to BaseStation in debug mode
        self.mission_success = False
        self.mission_max_status = self.mission_Not_end
        self.mission_duration = 0  # T_M in paper. unit: round
        self.mission_duration_max = mission_duration_max    # unit: round, initialize to 150
        self.mission_condition = 0  # -1 means game not end, 0 means mission success, 1 means mission failed by unfinished scan, 2, failed by no time, 3. failed by no MD
        self.map_cell_number = map_cell_number #5    # number of cell each side
        self.cell_size = 150 #100  # original 100 in meter     (left bottom point of cell represents the coordinate of cell
        self.map_size = self.map_cell_number * self.cell_size   # in meter
        self.map_ori_x = 1  # original point of target area
        self.map_ori_y = 1
        self.scan_map = np.zeros((self.map_size, self.map_size))
        # self.scan_cell_map = self.scan_map[::self.cell_size, ::self.cell_size]
        self.scan_rate = 0.1
        self.min_scan_requirement = 2.0 #0.99
        self.recalc_trajectory = False  # True: need recalculate trajectory.
        self.num_MD = 15 # number of MD (in index, MD first then HD)
        self.num_HD = num_HD # 2 # number of HD
        # self.MD_dict = {}  # key is id, value is class detail
        # self.HD_dict = {}
        self.Drone_dict = {}
        self.RLD = RLD_Drone(self.num_MD + self.num_HD)
        self.fleet_G = self.create_drone_fleet(self.num_MD, self.num_HD)
        self.MD_crashed_IDs = []  # includes MD crashed or crashed
        # self.assign_HD()
        # self.assign_MD()
        self.assign_Drone_dict()
        self.init_neighbor_table()
        self.update_neighbor_table()
        self.update_RLD_connection()

    @property
    def scan_cell_map(self):
        return self.scan_map[::self.cell_size, ::self.cell_size]

    def create_drone_fleet(self, MD_num, HD_num):
        '''
        ID start from RLD, then HD, then MD
        :param MD_num:
        :param HD_num:
        :return:
        '''

        fleet_G = nx.complete_graph(HD_num + MD_num)


        # assign honey drone to nodes
        for HD_id in range(HD_num):
            fleet_G.nodes[HD_id]['class'] = Honey_Drone(HD_id)
        # assign mission drone to nodes
        for MD_id in range(HD_num, HD_num+MD_num):
            fleet_G.nodes[MD_id]['class'] = Mission_Drone(MD_id)

        for node_id in range(fleet_G.number_of_nodes()):
            fleet_G.nodes[node_id]['pos'] = np.zeros(2)

        return fleet_G


    def assign_Drone_dict(self):
        for MD_id, MD in self.MD_dict.items():
            self.Drone_dict[MD_id] = MD
        for HD_id, HD in self.HD_dict.items():
            self.Drone_dict[HD_id] = HD
        self.Drone_dict[self.RLD.ID] = self.RLD

    @property
    def MD_dict(self):
        return {node_id: node for node_id, node in self.fleet_G.nodes(data='class') if node.type == 'MD'}

    # MD_set only contain MD that not crash
    @property
    def MD_set(self):
        return [MD for MD in self.MD_dict.values() if not MD.crashed]

    @property
    def MD_set_include_creashed(self):
        return [MD for MD in self.MD_dict.values()]

    # MD_connected_set only contain MD that not crash and connected to the RLD.
    # @property
    # def MD_connected_set(self):
    #     return [MD for MD in self.MD_dict.values() if not MD.crashed and MD.connect_RLD]
    @property
    def MD_connected_set(self):
        return [MD for MD in self.MD_dict.values() if not MD.crashed and MD.connect_RLD]

    @property
    def HD_dict(self):
        return {node_id: node for node_id, node in self.fleet_G.nodes(data='class') if node.type == 'HD'}

    # set of MD in mission (not in GCS)
    @property
    def MD_mission_set(self):
        return [MD for MD in self.MD_dict.values() if not MD.crashed and not MD.in_GCS]

    # HD_set only contain HD that not crash
    @property
    def HD_set(self):
        return [HD for HD in self.HD_dict.values() if not HD.crashed]

    @property
    def HD_set_include_creashed(self):
        return [HD for HD in self.HD_dict.values()]

    # HD_connected_set only contain HD that not crash
    @property
    def HD_connected_set(self):
        return [HD for HD in self.HD_dict.values() if not HD.crashed and HD.connect_RLD]

    # set of HD in mission (not in GCS)
    @property
    def HD_mission_set(self):
        return [HD for HD in self.HD_dict.values() if not HD.crashed and not HD.in_GCS]

    # RLD_set for coding convinience
    @property
    def RLD_set(self):
        return [self.RLD]

    # Drone_set only contain Drone that not crash
    @property
    def Drone_set(self):
        return self.HD_set + self.MD_set + self.RLD_set

    @property
    def Drone_set_include_creashed(self):
        return self.HD_set_include_creashed + self.MD_set_include_creashed + self.RLD_set

    # distance between two drones. (Eq. \eqref(Eq: distance) in paper)
    def drones_distance(self, drone_x_location, drone_y_location):
        distance_squre = np.square(drone_x_location - drone_y_location)
        return np.sqrt(np.sum(distance_squre))

    # iterate through all MD/HD and check their state (only consider alive drone)
    def Drone_state_update(self):
        if self.mission_duration > 1:
            for MD in self.MD_set:
                MD.condition_check()
            for HD in self.HD_set:
                HD.condition_check()

    # input type: np.ndarray
    def calc_distance(self, xyz1: np.ndarray, xyz2: np.ndarray):
        return np.linalg.norm(xyz1 - xyz2)

    # return integer as signal, round down decimal points
    # if value greater than 10, return 10
    def observed_signal(self, original_signal, distance):
        res = original_signal - 4 * 10 * np.log10(distance)
        return res

    # the maximum range of a drone under given transmitted signal strength
    # -100 dBm usually treated as the minimum valid signal strength.
    def signal_range(self, original_signal):
        res = 10 ** ((original_signal + 100) / 40)
        return res



    # scan count, and check if crashed
    def MD_environment_interaction(self):
        for MD in self.MD_set:
            if MD.crashed:
                continue

            xyz_current = MD.xyz_temp
            cell_x, cell_y, height_z = xyz_current
            # if new drone crashed，recalculate trajectory
            if MD.new_crash(xyz_current):
                self.recalc_trajectory = True   # when a MD offline, recalculate trajectory

            # update scanned map
            map_x_index = round(cell_x - 1)     # minus 1 for ignoring GCS
            map_y_index = round(cell_y - 1)     # minus 1 for ignoring GCS
            # map_size_with_station = self.map_size + 1

            # if drone's memory is not full, it can scan
            if not MD.memory_full:
                # if drone in target area
                if map_x_index in range(self.map_size) and map_y_index in range(self.map_size):
                    if MD.connect_RLD:          # scan only happen when connected to RLD.
                        self.scan_map[map_x_index, map_y_index] += self.scan_rate
            else:
                # if drone's memory is full, it can not scan. And reset memory_full flag to False
                MD.memory_full -= 1



    def HD_environment_interaction(self):
        for HD in self.HD_set:
            if HD.crashed:
                continue

            xyz_current = HD.xyz_temp

            # check if new drone crashed
            if HD.new_crash(xyz_current):      # when a HD crashed, no trajectory recalculate need
                pass


    def not_scanned_map(self):
        return self.scan_map < self.min_scan_requirement

    def print_MDs(self):
        for MD in self.MD_dict.values():
            print(MD)

    def print_HDs(self):
        for HD in self.HD_dict.values():
            print(HD)

    def print_system(self):
        print(vars(self))

    def print_drones_battery(self):
        for MD in self.MD_set:
            print(f"MD {MD.ID} battery: {MD.battery}")
        for HD in self.HD_set:
            print(f"HD {HD.ID} battery: {HD.battery}")

    def location_backup_MD(self):
        pass

    def battery_consume(self):
        # avoid recalc when unused drone in base station. Only recalc when: 1. drone crashed. 2. drone low battery. 3.....
        for MD in self.MD_set:
            if MD.battery_update():
                if self.print: print(f"detected MD {MD.ID} charging complete, recalculate trajectory")
                self.recalc_trajectory = True

        for HD in self.HD_set:  # True means battery charging complete
            if HD.battery_update():
                if self.print: print(f"detected HD {HD.ID} charging complete, recalculate trajectory")
                self.recalc_trajectory = True

    def HD_one_round_consume(self):
        total_consum = 0
        for HD in self.HD_set:
            if not HD.in_GCS:
                if not HD.crashed:
                    total_consum += HD.consume_rate
        return total_consum

    def MD_one_round_consume(self):
        total_consum = 0
        for MD in self.MD_set:
            if not MD.in_GCS:
                if not MD.crashed:
                    total_consum += MD.consume_rate
        return total_consum


    def assign_HD(self):
        for index in range(self.num_HD):
            self.HD_dict[index] = Honey_Drone(index, self.update_freq)

    def assign_MD(self):
        for index in range(self.num_MD):
            self.MD_dict[index + self.num_HD] = Mission_Drone(index + self.num_HD, self.update_freq)

    def init_neighbor_table(self):
        for drone in self.HD_set + self.MD_set:
            for id in range(self.num_MD + self.num_HD):
                drone.neighbor_table[id] = True
            drone.neighbor_table[self.num_MD + self.num_HD] = True     # this oen is for adding RLD

    def update_neighbor_table(self):
        for drone in self.HD_set + self.MD_set:
            for neighbor in self.HD_set + self.MD_set + self.RLD_set: # add RLD to calculate if in the range of RLD
                temp_dist = self.calc_distance(drone.xyz_temp, neighbor.xyz_temp)
                sign_range = self.signal_range(drone.signal)    # only consider the signal of the drone that send data.
                if temp_dist > sign_range:
                    drone.neighbor_table[neighbor.ID] = False
                else:
                    drone.neighbor_table[neighbor.ID] = True


        return

    def update_RLD_connection(self):
        for drone in self.HD_set + self.MD_set:
            self.init_visited()     # make all visited be False for DFS algorithm
            res = self.DFS_find_RLD(drone)
            # if not res:
            #     print("False Connect", drone.ID, drone.xyz_temp)
            drone.connect_RLD = res

    def init_visited(self):
        for drone in self.HD_set + self.MD_set:
            drone.visited = False


    def DFS_find_RLD(self, drone):
        drone.visited = True
        neighbor_list = self.connected_neighbors(drone)
        for neighbor in neighbor_list:
            if neighbor.type == 'RLD':
                return True
            else:
                if neighbor.visited is False:
                    res = self.DFS_find_RLD(neighbor)
                    if res is True:
                        return True
        return False


    def connected_neighbors(self, drone):
        res = []
        for neigh_id, neigh_state in drone.neighbor_table.items():  # neigh_state means connected or not.
            if neigh_state is True:
                if not self.Drone_dict[neigh_id].crashed:
                    res.append(self.Drone_dict[neigh_id])
        return res



    def sample_MD(self):
        return Mission_Drone(-1, self.update_freq)

    def aliveDroneCount(self):
        return len(self.MD_set) + len(self.HD_set)

    @property
    def sample_HD(self):
        return Honey_Drone(-2, self.update_freq)

    # this is system information update function called once per round
    def check_mission_complete(self):
        # update system information
        self.mission_duration += 1

        # check if mission can end
        # scan_cell_map = self.scan_map[::self.cell_size, ::self.cell_size]   # transfer meter-based scan map to cell-based scan map
        if np.all(self.scan_cell_map >= self.min_scan_requirement):
            # mission end if: all cells are scanned
            self.mission_Not_end -= 1
            self.mission_success = True
        elif self.mission_duration > self.mission_duration_max:
            # mission end if: mission time limit meet
            self.mission_Not_end -= 2
            print("\n Mission Fail :( \n")
            self.mission_condition = 2
        else:
            pass
            # self.print_drones_battery()

        # check if mission end successfully
        if self.mission_Not_end <= 0:
            if self.mission_success:
                print("\n Mission Complete!! \n")
                self.mission_condition = 0
                if self.print: print("cell-based scanned map:\n", self.scan_cell_map)
                if self.print: self.print_MDs()
                if self.print: self.print_HDs()
                if self.print: self.print_system()
            else:
                print("\n Mission Fail :( \n")
                self.mission_condition = 1
                if self.print: print("cell-based scanned map:\n", self.scan_cell_map)
                if self.print: self.print_MDs()
                if self.print: self.print_HDs()
                if self.print: self.print_system()

        return self.mission_Not_end

    # this is a fast way to check for saving running time
    def is_mission_Not_end(self):
        return self.mission_Not_end > 0

    def scanCompletePercent(self):
        # scan_cell_map = self.scan_map[::self.cell_size,::self.cell_size]  # transfer meter-based scan map to cell-based scan map
        scan_complete_map = self.scan_cell_map >= self.min_scan_requirement
        count_true = np.count_nonzero(scan_complete_map)
        return count_true/self.scan_cell_map.size

    def scanCompleteCount(self):
        # scan_cell_map = self.scan_map[::self.cell_size,::self.cell_size]  # transfer meter-based scan map to cell-based scan map
        scan_complete_map = self.scan_cell_map >= self.min_scan_requirement
        count_true = np.count_nonzero(scan_complete_map)
        return count_true

    def scanCellMapSum(self):
        return np.sum(self.scan_cell_map)

    def update_scan(self, x_axis, y_axis, amount):
        x_axis = x_axis - self.map_ori_x
        y_axis = y_axis - self.map_ori_y
        if 0 <= x_axis and x_axis < self.map_size and 0 <= y_axis and y_axis < self.map_size:
            self.scan_map[x_axis, y_axis] += amount

    def draw_MD_Position(self):
        pos_dict = {}
        for drone_id, drone_attribute in self.fleet_G.nodes(data=True):
            drone = drone_attribute['class']
            pos_dict[drone.ID] = drone.xyz_temp[:2]

        nx.draw(self.fleet_G, pos_dict, with_labels=True)
        plt.show()