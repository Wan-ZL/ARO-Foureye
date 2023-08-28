import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from scipy.stats import norm
import statistics
import sys


schemes = ["DD-PI", "DD-IPI", "DD-ML-IPI"]
schemes_index = 2
path = "data/trainning_data/" + schemes[schemes_index]
file_list = [f for f in os.listdir(path) if not f.startswith('.')]
file_name = file_list[0]
the_file = open("data/trainning_data/" + schemes[schemes_index] + "/" + file_name, "rb")
all_result = pickle.load(the_file)
# print(all_result_after_each_game_all_result)
the_file.close()
# print(np.round(all_result[0],1))

all_result_concat = np.array(all_result[0])
for simulation_index in range(1, len(all_result)):
    all_result_concat = np.concatenate((all_result_concat, np.array(all_result[simulation_index])), axis=0)

# granu = 10
# counter = np.zeros((9,11)) # 9 strategy, 11 class
# for simulation_index in range(len(all_result)):
#     for game_index in range(len(all_result[simulation_index])):
#         for strategy_index in range(9):
#             # print(all_result[simulation_index][game_index][strategy_index])
#             # print(round(all_result[simulation_index][game_index][strategy_index],1))
#             counter[strategy_index, int(round(all_result[simulation_index][game_index][strategy_index],1)*granu)] += 1
# # print(int(all_result[0][7][0]*10))
#
#
# # Calculating mean and standard deviation
all_result_concat = np.round(all_result_concat, 5)
for index in range(all_result_concat.shape[1]):
    sorted_array = np.sort(all_result_concat[:,index])
    print(index)
    # print(sorted_array)
    print(sorted_array.shape)
    mean = statistics.mean(sorted_array)
    print(f"mean {mean}")
    sd = statistics.stdev(sorted_array)
    print(f"sd {sd}")

    plt.plot(sorted_array, norm.pdf(sorted_array, mean, sd), label=f"strategy {index+1}")
plt.legend()
plt.xlabel("S_j")
plt.ylabel("PDF")
plt.show()
