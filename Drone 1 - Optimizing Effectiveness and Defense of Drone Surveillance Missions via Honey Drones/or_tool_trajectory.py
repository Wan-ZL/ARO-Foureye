
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random
import math
import time


def create_data_model(map_cell_number, cell_size, locations_MD, num_vehicles, depot, not_scanned_map):
    """Stores the data for the problem."""
    data = {}
    data['num_vehicles'] = num_vehicles    # vehicle/drone number
    data['depot'] = depot           # base station index
    data['map_cell_number'] = map_cell_number
    data['cell_size'] = cell_size

    baseStation_x = 0
    baseStation_y = 0
    cell_number = map_cell_number * map_cell_number + 1  # including base station
    map_x_start = baseStation_x + 1
    map_x_end = map_cell_number * cell_size +1
    map_y_start = baseStation_y + 1
    map_y_end = map_cell_number * cell_size + 1

    # distances
    # In this test file, locations are tuple
    location_set = [(baseStation_x, baseStation_y)]
    for x in range(map_x_start, map_x_end, cell_size):
        for y in range(map_y_start, map_y_end, cell_size):
            # trajectory only consider cells not scanned, not_scanned_map index start from 0
            if not_scanned_map[x-map_x_start, y-map_y_start]:
                location_set.append((x, y))
    # add drones location
    for location in locations_MD:
        if location not in location_set:
            location_set.append(location)
    data['locations'] = location_set

    # data['distance_matrix'], data['node2position'], data['position2node'] = compute_euclidean_distance_matrix(data['locations'],)
    # # link position xy with node index
    # data['position2node'] = {}
    # data['node2position'] = {}
    # for from_counter, from_node in enumerate(data['locations']):
    #     data['node2position'][from_counter] = from_node
    #     data['position2node'][from_node] = from_counter

    # calculate distance
    # distance_set = np.zeros((cell_number, cell_number))
    # # print_debug('shape', distance_set.shape)
    # # print_debug("distance_set", distance_set)
    # i_index = 0
    # for i_x, i_y in location_set:
    #     j_index = 0
    #     for j_x, j_y in location_set:
    #         # if not_scanned_map[i_index, j_index]:  # trajectory only consider cells not scanned
    #         distance_set[i_index][j_index] = np.linalg.norm([i_x - j_x, i_y - j_y])
    #         j_index += 1
    #     i_index += 1
    #
    # # since the libarary will convert float to integer, Multiple 100 here for keeping two decimals.
    # data['distance_matrix'] = distance_set * 100
    return data

def compute_euclidean_distance_matrix(locations, variable_precision):
    """Creates callback to return distance between points."""
    distances = {}
    node2position = {}
    position2node = {}
    for from_counter, from_node in enumerate(locations):
        distances[from_counter] = {}
        node2position[from_counter] = from_node
        position2node[from_node] = from_counter
        for to_counter, to_node in enumerate(locations):
            if from_counter == to_counter:
                distances[from_counter][to_counter] = 0
            else:
                # Euclidean distance
                distances[from_counter][to_counter] = (int(
                    math.hypot((from_node[0] - to_node[0]),
                               (from_node[1] - to_node[1]))*variable_precision))   # multiple variable_precision for keeping two decimals
    return distances, node2position, position2node



def print_solution(data, manager, routing, solution, variable_precision, node2position):
    """Prints solution on console."""
    print(f'Objective: {solution.ObjectiveValue()/variable_precision}')
    max_route_distance = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0
        previous_index = index
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            plan_output += f' {node} {node2position[node]} -> '
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
            # print_debug("distance", manager.IndexToNode(index), manager.IndexToNode(previous_index), data['distance_matrix'][manager.IndexToNode(index)][manager.IndexToNode(previous_index)])
        node = manager.IndexToNode(index)
        plan_output += f' {node} {node2position[node]}\n'
        route_distance += routing.GetArcCostForVehicle(
            previous_index, index, vehicle_id)
        plan_output += 'Distance of the route: {}m\n'.format(route_distance/variable_precision)
        print(plan_output)
        max_route_distance = max(route_distance, max_route_distance)
    print('Maximum of the route distances: {}m\n'.format(max_route_distance/variable_precision))


def draw_map(data, manager, routing, solution):
    fig, ax = plt.subplots()
    location_set = data['locations']
    map_cell_number = data['map_cell_number']
    cell_size = data['cell_size']
    map_size = map_cell_number * cell_size
    colors_order = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors_len = len(colors_order) - 1

    # draw grid
    plt.plot([0.5, 0.5], [-0.5, map_size + 0.5], linewidth=1, color='gray', zorder=0)
    plt.plot([-0.5, map_size + 0.5], [0.5, 0.5], linewidth=1, color='gray', zorder=0)

    # draw meter-unit-based grid
    # if cell_size <= 20:      # takes too many time to draw meter unit lines. So ignore this when cell size too large
    for i in range(1,map_size):
        plt.plot([i+0.5, i+0.5], [0.5, map_size+0.5], linewidth=0.2, color='gray', zorder=0)
        plt.plot([0.5, map_size + 0.5], [i + 0.5, i + 0.5], linewidth=0.2, color='gray', zorder=0)
        # plt.axvline(x=i + 0.5, linewidth=0.5, color='gray', zorder=0)
        # plt.axhline(y=i + 0.5, linewidth=0.5, color='gray', zorder=0)

    # draw cell-based grid
    for i in range(1, map_size+cell_size, cell_size):
        plt.plot([i-0.5, i-0.5], [0.5, map_size+0.5], linewidth=0.5, color='gray', zorder=1)
        plt.plot([0.5, map_size + 0.5], [i - 0.5, i - 0.5], linewidth=0.5, color='gray', zorder=1)


    # draw gray to target area               # draw this first to make it on lowest layer.
    # if cell_size <= 20:  # takes too many time to draw patch. So ignore this
    ax.add_patch(Rectangle((-0.5, -0.5), 1, 1, color='0.95', zorder=0))
    ax.add_patch(Rectangle((0.5, 0.5), map_size, map_size, color='0.95', zorder=0))
        # for i in range(map_size):
        #     for j in range(map_size):
        #         ax.add_patch(Rectangle((i + 0.5, j + 0.5), 1, 1, color='0.95', zorder=0))


    for vehicle_id in range(data['num_vehicles']):
        # print_debug("vehicle_id", vehicle_id)
        # color = (random.random(), random.random(), random.random())

        index = routing.Start(vehicle_id)
        route_distance = 0

        [i_x, i_y] = [None, None]
        while not routing.IsEnd(index):
            # print_debug("IndexToNode", manager.IndexToNode(index))
            # print_debug("[i_x, i_y]", [i_x, i_y])
            [j_x, j_y] = location_set[manager.IndexToNode(index)]
            if i_x is not None:
                plt.arrow(i_x, i_y, j_x - i_x, j_y - i_y, width=0.05, color = colors_order[vehicle_id % colors_len], length_includes_head=True)
            [i_x, i_y] = [j_x, j_y]
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
        # print_debug("IndexToNode", manager.IndexToNode(index))
        # print_debug("[i_x, i_y]", [i_x, i_y])
        [j_x, j_y] = location_set[manager.IndexToNode(index)]
        if i_x is not None:
            plt.arrow(i_x, i_y, j_x - i_x, j_y - i_y, width=0.05, color = colors_order[vehicle_id % colors_len], length_includes_head=True)
        plt.plot(j_x, j_y, 'ro')
        # print_debug(solution.Value(routing.NextVar(index)))
        # print_debug("route_distance", route_distance)

    # plt.plot([0.5, 0.5], [0.5, 4.5], color='gray')

    plt.xlim(-0.5, map_size + 0.5)
    plt.ylim(-0.5, map_size + 0.5)
    plt.show()


# In some case, alive drone indexs are 1,3,6. Use 'indexs_MD' for assign accurate index name
def generate_route_array(data, manager, routing, solution, indexs_MD):
    location_set = data['locations']
    route_array = {}
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        route_array[indexs_MD[vehicle_id]] = np.array([location_set[manager.IndexToNode(index)]]).astype(float)
        while not routing.IsEnd(index):
            index = solution.Value(routing.NextVar(index))
            route_array[indexs_MD[vehicle_id]] = np.concatenate((route_array[indexs_MD[vehicle_id]],[location_set[manager.IndexToNode(index)]]), axis=0)


        # route_array[vehicle_id] = route_array[vehicle_id][1:]

    return route_array


# def fix_result():
#     res = {2: np.array([[0., 0.],
#                         [401., 1.],
#                         [301., 1.],
#                         [201., 1.],
#                         [101., 1.],
#                         [0., 0.]]), 3: np.array([[0., 0.],
#                                                  [101., 301.],
#                                                  [101., 401.],
#                                                  [201., 401.],
#                                                  [301., 401.],
#                                                  [201., 301.],
#                                                  [101., 201.],
#                                                  [0., 0.]]), 4: np.array([[0., 0.],
#                                                                           [1., 1.],
#                                                                           [401., 401.],
#                                                                           [301., 301.],
#                                                                           [201., 201.],
#                                                                           [101., 101.],
#                                                                           [0., 0.]]), 5: np.array([[0., 0.],
#                                                                                                    [301., 101.],
#                                                                                                    [401., 101.],
#                                                                                                    [401., 201.],
#                                                                                                    [401., 301.],
#                                                                                                    [301., 201.],
#                                                                                                    [201., 101.],
#                                                                                                    [0., 0.]]),
#            6: np.array([[0., 0.],
#                         [1., 401.],
#                         [1., 301.],
#                         [1., 201.],
#                         [1., 101.],
#                         [0., 0.]])}
#     return res


def MD_path_plan_main(indexs_MD, locations_MD, map_cell_number, cell_size, not_scanned_map):
    print_traj = False
    """Entry point of the program."""
    # print_debug("locations_MD", locations_MD)
    # map_cell_number = 5
    num_vehicles = len(locations_MD)  # vehicle/drone number
    depot = 0  # base station index
    variable_precision = 100 # 100 means keep two decimal

    # Instantiate the data problem.
    data = create_data_model(map_cell_number, cell_size, locations_MD, num_vehicles, depot, not_scanned_map)

    # create distance for all nodes.
    distance_matrix, node2position, position2node = compute_euclidean_distance_matrix(data['locations'], variable_precision)
    # start & end location
    data['starts'] = []
    for x, y in locations_MD:
        posi_index = position2node[(x,y)]

        data['starts'].append(posi_index)

    data['ends'] = [depot] * num_vehicles

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['locations']), data['num_vehicles'], data['starts'], data['ends'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]
        # return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Distance constraint.
    map_size = map_cell_number * cell_size
    dimension_name = 'Distance'
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        (map_size*variable_precision)*(map_size*variable_precision),  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(map_size*variable_precision)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC) # algorithm may change here)

    # Setting second solution
    # search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    # search_parameters.first_solution_strategy = (
    #     routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC)
    # search_parameters.local_search_metaheuristic = (
    #     routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC)
    # search_parameters.time_limit.seconds = 100
    # search_parameters.log_search = True

    # Solve the problem (with time counter).
    start = time.time()
    # This may cause the tread error (cannot create a new thread)
    solution = routing.SolveWithParameters(search_parameters)
    end = time.time()
    # print_debug("\ntime spent:{:.2f}\n".format(end - start))

    # Print solution on console.
    if solution:
        if print_traj: print_solution(data, manager, routing, solution, variable_precision, node2position)
        if print_traj: draw_map(data, manager, routing, solution)
    else:
        print('No solution found !')

    return generate_route_array(data, manager, routing, solution, indexs_MD)


if __name__ == '__main__':
    num_MD = 6
    index_MD = range(num_MD)
    # index_MD = [1,3,6]
    locations_MD = []
    for i in range(num_MD):
        locations_MD.append((0,0))
    map_cell_number = 10
    cell_size = 100
    map_size = map_cell_number * cell_size
    not_scanned_map = np.ones((map_size,map_size), dtype=bool)
    # not_scanned_map[0,cell_size] = False
    # not_scanned_map[cell_size, 0] = False
    MD_path_plan_main(index_MD, locations_MD, map_cell_number, cell_size, not_scanned_map)


