import random

import pulp
import numpy as np
import matplotlib.pyplot as plt
import re

def extractInt(the_String):
    int_set = [int(num) for num in re.findall(r"\d+", the_String)]
    return int_set

# extractInt("X_(4,4)(2,2)k")
# extractInt("y_(0,0)k")
# quit()

def draw_map(map_size, variables):
    if map_size <= 0:
        return

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    for v in MyProbLP.variables():
        v_name = v.name
        if v_name[0] == 'X':
            if v.varValue != 0:
                i_x, i_y, j_x, j_y = extractInt(v_name)
                color = (random.random(), random.random(), random.random())
                # plt.plot([i_x, j_x], [i_y, j_y], linewidth = 5)
                plt.arrow(i_x, i_y, j_x-i_x, j_y-i_y, width = 0.05, facecolor = color, length_includes_head=True)

    # draw grid
    for i in range(map_size+1):
        plt.axvline(x=i+0.5, linewidth=0.5, color='gray')
        plt.axhline(y=i + 0.5, linewidth=0.5, color='gray')

    plt.xlim(-0.5, map_size+1+0.5)
    plt.ylim(-0.5, map_size+1+0.5)

    plt.show()


if __name__ == '__main__':
    MyProbLP = pulp.LpProblem("MD_Demo", sense=pulp.LpMinimize)

    E_T = 1
    map_size = 7
    # E = {(0,0),(0,1),(1,0),(1,1)}
    #                   (i_x, i_y), (j_x, j_y)
    # d_ij = np.zeros((map_cell_number,map_cell_number,map_cell_number,map_cell_number))

    # Base station location (BS and cell locations cannot be the same)
    baseStation_x = map_size
    baseStation_y = map_size

    # distances
    d_ij = {}
    for i_x in range(map_size):
        for i_y in range(map_size):
            # cell to cell distance
            for j_x in range(map_size):
                for j_y in range(map_size):
                    if (i_x,i_y) != (j_x,j_y):
                        d_ij[(i_x, i_y), (j_x, j_y)] = np.linalg.norm([i_x - j_x, i_y - j_y])
            # base station to cell distances
            d_ij[(baseStation_x, baseStation_y), (i_x, i_y)] = np.linalg.norm([baseStation_x - i_x, baseStation_y - i_y])
            d_ij[(i_x, i_y), (baseStation_x, baseStation_y)] = np.linalg.norm([i_x - baseStation_x, i_y - baseStation_y])

    print("d_ij", d_ij)
    print("len(d_ij)", len(d_ij))



    # variables for edge
    X_ijk = {}
    for i_x in range(map_size):
        for i_y in range(map_size):
            # for cells
            for j_x in range(map_size):
                for j_y in range(map_size):
                    if (i_x, i_y) != (j_x, j_y):

                        X_ijk[(i_x,i_y),(j_x,j_y)] = pulp.LpVariable(f'X_({i_x},{i_y})({j_x},{j_y})k', lowBound=0, upBound=1, cat='Integer')
            # for base station to cell
            X_ijk[(baseStation_x, baseStation_y), (i_x, i_y)] = pulp.LpVariable(f'X_({baseStation_x},{baseStation_y})({i_x},{i_y})k', lowBound=0, upBound=1, cat='Integer')
            X_ijk[(i_x, i_y), (baseStation_x, baseStation_y)] = pulp.LpVariable(f'X_({i_x},{i_y})({baseStation_x},{baseStation_y})k', lowBound=0, upBound=1, cat='Integer')

    # print_debug(type(X_ijk[(0,0),(1,1)]))
    # test start
    # X_ijk[(2, 2), (1, 1)] = pulp.LpVariable(f'X_({2},{2})({1},{1})k', lowBound=1, upBound=1, cat='Integer')
    # X_ijk[(1, 1), (0, 1)] = pulp.LpVariable(f'X_({1},{1})({0},{1})k', lowBound=1, upBound=1, cat='Integer')
    # X_ijk[(0, 1), (0, 0)] = pulp.LpVariable(f'X_({0},{1})({0},{0})k', lowBound=1, upBound=1, cat='Integer')
    # X_ijk[(0, 0), (1, 0)] = pulp.LpVariable(f'X_({0},{0})({1},{0})k', lowBound=1, upBound=1, cat='Integer')
    # X_ijk[(1, 0), (2, 2)] = pulp.LpVariable(f'X_({1},{0})({2},{2})k', lowBound=1, upBound=1, cat='Integer')
    # test end
    print("X_ijk", X_ijk)
    print("len(X_ijk)",len(X_ijk))

    # variables for cell (node)
    y_ik = {}   # non-negative integer
    # for cells
    for i_x in range(map_size):
        for i_y in range(map_size):
            y_ik[(i_x,i_y)] = pulp.LpVariable(f'y_({i_x},{i_y})k', lowBound=0, cat='Integer')
    # for base station
    y_ik[(baseStation_x, baseStation_y)] = pulp.LpVariable(f'y_({baseStation_x},{baseStation_y})k', lowBound=0, cat='Integer')
    print("y_ik", y_ik)
    print("len(y_ik)",len(y_ik))

    # create target function
    sum_value = 0
    # MyProbLP += E_T * d_ij[0][0] * X_test_k
    all_variable = 0 #pulp.LpVariable('init', lowBound=0, upBound=0, cat='Integer')
    for i_x in range(map_size):
        for i_y in range(map_size):
            # cells to cells
            for j_x in range(map_size):
                for j_y in range(map_size):
                    if (i_x, i_y) != (j_x, j_y):
                        sum_value += E_T * d_ij[(i_x,i_y),(j_x,j_y)] * X_ijk[(i_x,i_y),(j_x,j_y)]
                        all_variable += X_ijk[(i_x,i_y),(j_x,j_y)]
            # base station to cells
            sum_value += E_T * d_ij[(baseStation_x, baseStation_y), (i_x, i_y)] * X_ijk[(baseStation_x, baseStation_y), (i_x, i_y)]
            sum_value += E_T * d_ij[(i_x, i_y), (baseStation_x, baseStation_y)] * X_ijk[(i_x, i_y), (baseStation_x, baseStation_y)]
            # all_variable += X_ijk[(baseStation_x, baseStation_y), (i_x, i_y)]
    print("sum_value", sum_value)
    # save target function
    MyProbLP += sum_value


    # constrain functions
    # Eq.(5)
    t_i = 1
    for i_x in range(map_size):
        for i_y in range(map_size):
            MyProbLP += (y_ik[(i_x,i_y)] == t_i)
    # MyProbLP += (y_ik[(baseStation_x, baseStation_y)] == t_i)
    # MyProbLP += (y_ik[(baseStation_x, baseStation_y)] == t_i) # extra for testing

    # Eq.(7)
    # for cells
    for i_x in range(map_size):
        for i_y in range(map_size):
            sum_X_ijk = 0
            for j_x in range(map_size):
                for j_y in range(map_size):
                    if (i_x, i_y) != (j_x, j_y):
                        sum_X_ijk += X_ijk[(i_x,i_y),(j_x,j_y)]
            sum_X_ijk += X_ijk[(i_x, i_y), (baseStation_x, baseStation_y)]
            MyProbLP += (y_ik[(i_x,i_y)] == t_i * sum_X_ijk)
    # for base station
    BS_sum_X_ijk = 0
    for j_x in range(map_size):
        for j_y in range(map_size):
            BS_sum_X_ijk += X_ijk[(baseStation_x, baseStation_y), (j_x, j_y)]
    MyProbLP += (y_ik[(baseStation_x, baseStation_y)] <= t_i * BS_sum_X_ijk)

    # Eq.(8)
    # for cells
    for i_x in range(map_size):
        for i_y in range(map_size):
            sum_X_ijk_l = 0
            sum_X_jik = 0
            for j_x in range(map_size):
                for j_y in range(map_size):
                    if (i_x, i_y) != (j_x, j_y):
                        # X_ij
                        sum_X_ijk_l += X_ijk[(i_x,i_y), (j_x,j_y)]
                        # X_ji
                        sum_X_jik += X_ijk[(j_x,j_y), (i_x,i_y)]
            # X_ij (includes BS)
            sum_X_ijk_l += X_ijk[(i_x, i_y), (baseStation_x, baseStation_y)]
            # X_ji (includes BS)
            sum_X_jik += X_ijk[ (baseStation_x, baseStation_y), (i_x, i_y)]
            MyProbLP += (sum_X_ijk_l == sum_X_jik)
    # for base station
    BS_sum_X_ijk = 0
    BS_sum_X_jik = 0
    for j_x in range(map_size):
        for j_y in range(map_size):
            # X_ij
            BS_sum_X_ijk += X_ijk[(baseStation_x, baseStation_y), (j_x, j_y)]
            # X_ji
            BS_sum_X_jik += X_ijk[(j_x, j_y), (baseStation_x, baseStation_y)]
    MyProbLP += (BS_sum_X_ijk == BS_sum_X_jik)

    # Eq.(9)
    sum_0jk = 0
    for j_x in range(map_size):
        for j_y in range(map_size):
            sum_0jk += X_ijk[(baseStation_x, baseStation_y), (j_x, j_y)]
    MyProbLP += (sum_0jk == 1)

    # all drone must back to station
    # sum_i0k = 0
    # for i_x in range(map_cell_number):
    #     for i_y in range(map_cell_number):
    #         sum_i0k += X_ijk[(j_x, j_y), (baseStation_x, baseStation_y)]
    # MyProbLP += (sum_i0k == 1)

    # Eq.(10)
    # sum_sum_X_ijk = 0
    # for i_x in range(map_cell_number):
    #     for i_y in range(map_cell_number):
    #         for j_x in range(map_cell_number):
    #             for j_y in range(map_cell_number):
    #                 if (i_x, i_y) != (j_x, j_y):
    #                     sum_sum_X_ijk += X_ijk[(i_x,i_y),(j_x,j_y)]
    # S_1 = map_cell_number * map_cell_number - 1
    # MyProbLP += (sum_sum_X_ijk <= S_1)
    # MyProbLP += (all_variable <= (map_cell_number*map_cell_number-1)) # Error: should be '>=' or '<='?

    # extra constrain
    for i_x in range(map_size):
        for i_y in range(map_size):
            MyProbLP += (X_ijk[(i_x, i_y), (baseStation_x, baseStation_y)] + X_ijk[(baseStation_x, baseStation_y), (i_x, i_y)] <= 1)
            for j_x in range(map_size):
                for j_y in range(map_size):
                    if (i_x, i_y) != (j_x, j_y):
                        MyProbLP += (X_ijk[(i_x,i_y),(j_x,j_y)]+X_ijk[(j_x,j_y),(i_x,i_y)] <= 1)
    # test start
    # MyProbLP += (X_ijk[(2, 2), (1, 1)] == 1)
    # MyProbLP += (X_ijk[(1, 1), (0, 1)] == 1)
    # MyProbLP += (X_ijk[(0, 1), (0, 0)] == 1)
    # MyProbLP += (X_ijk[(0, 0), (1, 0)] == 1)
    # MyProbLP += (X_ijk[(1, 0), (2, 2)] == 1)
    # MyProbLP += (X_ijk[(1, 0), (2, 2)] == 1)
    # test end

    print("\nStart solving:")
    MyProbLP.solve()
    draw_map(map_size, MyProbLP.variables())
    # print_debug("MyProbLP.variables()",MyProbLP)
    # for v in MyProbLP.variables():
    #     print_debug(v.name, "=", v.varValue)
    print("F(x) = ", pulp.value(MyProbLP.objective))


