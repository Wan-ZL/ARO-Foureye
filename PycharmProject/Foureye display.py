import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import StrMethodFormatter




# read data


#
# current_scheme = "DD-PI"
# the_file = open("data/"+current_scheme+"/R3/defender_uncertainty.pkl", "rb")
# def_uncertainty_history = pickle.load(the_file)
# the_file.close()
#
# the_file = open("data/"+current_scheme+"/R3/attacker_uncertainty.pkl", "rb")
# att_uncertainty_history = pickle.load(the_file)
# the_file.close()
#
# the_file = open("data/"+current_scheme+"/Time_to_SF.pkl", "rb")
# Time_to_SF = pickle.load(the_file)
# the_file.close()




# import pylab

# schemes = ["DD-IPI", "DD-PI", "No-DD-IPI", "No-DD-PI"]
# fig = pylab.figure()
# figlegend = pylab.figure(figsize=(3, 2))
# ax = fig.add_subplot(111)
# for index in range(4):
#     lines = ax.plot(range(10), pylab.randn(10))
#     figlegend.legend(lines, [schemes[index]], 'center')
# # lines = ax.plot(range(10), pylab.randn(10), range(10), pylab.randn(10),
# #                 range(10), pylab.randn(10), range(10), pylab.randn(10))

# fig.show()
# figlegend.show()
# figlegend.savefig('legend.png')




# Attacker's HEU

def display_att_HEU(schemes):
    plt.figure(figsize=(figure_width, figure_high + 0.75))
    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/R1/att_HEU.pkl", "rb")
        att_HEU_history = pickle.load(the_file)
        print(att_HEU_history)
        the_file.close()

        max_length = 0
        for key in att_HEU_history.keys():
            if max_length < len(att_HEU_history[key]):
                max_length = len(att_HEU_history[key])

        average_att_HEU = []
        for index in range(max_length):
            sum_on_index = 0
            number_on_index = 0
            for key in att_HEU_history.keys():
                if len(att_HEU_history[key]) > 0:
                    sum_on_index += att_HEU_history[key][0]
                    att_HEU_history[key].pop(0)
                    number_on_index += 1
            average_att_HEU.append(sum_on_index / number_on_index)

        x_values = range(len(average_att_HEU))
        y_values = average_att_HEU
        plt.plot(x_values[1:],
                 y_values[1:],
                 linestyle=all_linestyle[schemes_index],
                 label=schemes[schemes_index])
    plt.legend(prop={"size": legend_size},
               ncol=4,
               bbox_to_anchor=(-0.13, 1, 1.15, 0),
               loc='lower left',
               mode="expand")
    plt.xlabel("number of games", fontsize=font_size)
    plt.ylabel("Attacker HEU", fontsize=font_size)
    plt.xticks(fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    plt.tight_layout()
    plt.savefig("Figure/att-HEU-NG.eps", dpi=figure_dpi)


# Defender's HEU
def display_def_HEU(schemes):
    plt.figure(figsize=(figure_width, figure_high + 0.75))
    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/R1/def_HEU.pkl", "rb")
        def_HEU_history = pickle.load(the_file)
        the_file.close()

        max_length = 0
        for key in def_HEU_history.keys():
            if max_length < len(def_HEU_history[key]):
                max_length = len(def_HEU_history[key])

        average_def_HEU = []
        for index in range(max_length):
            sum_on_index = 0
            number_on_index = 0
            for key in def_HEU_history.keys():
                if len(def_HEU_history[key]) > 0:
                    sum_on_index += def_HEU_history[key][0]
                    def_HEU_history[key].pop(0)
                    number_on_index += 1
            average_def_HEU.append(sum_on_index / number_on_index)

        x_values = range(len(average_def_HEU))
        y_values = average_def_HEU
        plt.plot(x_values[1:], y_values[1:], linestyle=all_linestyle[schemes_index], label=schemes[schemes_index])
    plt.legend(prop={"size": legend_size},
               ncol=4,
               bbox_to_anchor=(-0.13, 1, 1.15, 0),
               loc='lower left',
               mode="expand")
    plt.xlabel("number of games", fontsize=font_size)
    plt.ylabel("Defender HEU", fontsize=font_size)
    plt.xticks(fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    plt.tight_layout()
    plt.savefig("Figure/def-HEU-NG.eps", dpi=figure_dpi)


# Attacker Cost
def display_att_cost(schemes):
    plt.figure(figsize=(figure_width, figure_high + 0.75))
    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/R6/att_cost.pkl", "rb")
        att_cost_all_result = pickle.load(the_file)
        the_file.close()

        max_length = 0
        for key in att_cost_all_result.keys():
            if max_length < len(att_cost_all_result[key]):
                max_length = len(att_cost_all_result[key])

        average_att_cost = []
        for index in range(max_length):
            sum_on_index = 0
            number_on_index = 0
            for key in att_cost_all_result.keys():
                if len(att_cost_all_result[key]) > 0:
                    sum_on_index += att_cost_all_result[key][0]
                    att_cost_all_result[key].pop(0)
                    number_on_index += 1
            average_att_cost.append(sum_on_index / number_on_index)

        x_values = range(len(average_att_cost))
        y_values = average_att_cost

        plt.plot(x_values[1:], y_values[1:], linestyle=all_linestyle[schemes_index], label=schemes[schemes_index])
    plt.legend(prop={"size": legend_size},
               ncol=4,
               bbox_to_anchor=(-0.18, 1, 1.2, 0),
               loc='lower left',
               mode="expand")
    plt.xlabel("number of games", fontsize=font_size)
    plt.ylabel("Attacker cost", fontsize=font_size)
    plt.xticks(fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    plt.tight_layout()
    plt.savefig("Figure/att-cost-NG.eps", dpi=figure_dpi)
    plt.show()


# Defender Cost
def display_def_cost(schemes):
    plt.figure(figsize=(figure_width, figure_high + 0.75))
    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/R6/def_cost.pkl", "rb")
        def_cost_all_result = pickle.load(the_file)
        the_file.close()

        max_length = 0
        for key in def_cost_all_result.keys():
            if max_length < len(def_cost_all_result[key]):
                max_length = len(def_cost_all_result[key])

        average_def_cost = []
        for index in range(max_length):
            sum_on_index = 0
            number_on_index = 0
            for key in def_cost_all_result.keys():
                if len(def_cost_all_result[key]) > 0:
                    sum_on_index += def_cost_all_result[key][0]
                    def_cost_all_result[key].pop(0)
                    number_on_index += 1
            average_def_cost.append(sum_on_index / number_on_index)

        x_values = range(len(average_def_cost))
        y_values = average_def_cost

        plt.plot(x_values[1:], y_values[1:], linestyle=all_linestyle[schemes_index], label=schemes[schemes_index])
    plt.legend(prop={"size": legend_size},
               ncol=4,
               bbox_to_anchor=(-0.18, 1, 1.2, 0),
               loc='lower left',
               mode="expand")
    plt.xlabel("number of games", fontsize=font_size)
    plt.ylabel("defender cost", fontsize=font_size)
    plt.xticks(fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    plt.tight_layout()
    plt.savefig("Figure/def-cost-NG.eps", dpi=figure_dpi)


# Attacker Strategy DD-IPI
def DD_IPI_att_strat_count():
    the_file = open("data/DD-IPI/R2/att_strategy_counter.pkl", "rb")
    att_strat_count = pickle.load(the_file)
    the_file.close()

    plt.figure(figsize=(figure_width, figure_high + 1))
    max_length = 0
    for key in att_strat_count.keys():
        if max_length < len(att_strat_count[key]):
            max_length = len(att_strat_count[key])

    average_att_strat = []
    for index in range(max_length):
        sum_on_index = 0
        number_on_index = 0
        att_strat = np.zeros(8)
        for key in att_strat_count.keys():
            if len(att_strat_count[key]) > 0:
                att_strat[att_strat_count[key][0]] += 1
                att_strat_count[key].pop(0)
                number_on_index += 1
        average_att_strat.append((att_strat / number_on_index).tolist())
    average_att_strat = np.array(average_att_strat)

    for index in range(8):
        plt.plot(range(max_length),
                 average_att_strat[:, index],
                 linestyle=all_linestyle[index],
                 label=f"Stra {index + 1}")

    plt.legend(prop={"size": legend_size},
               ncol=4,
               bbox_to_anchor=(0, 1, 1, 0),
               loc='lower left',
               mode="expand")
    plt.xlabel("number of games", fontsize=font_size)
    plt.ylabel("$P_S^A$", fontsize=font_size)
    plt.xticks(fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    plt.tight_layout()
    plt.savefig("Figure/DD-IPI-att-Strat.eps", dpi=figure_dpi)


# Defender Strategy DD-IPI
def DD_IPI_def_strat_count():
    the_file = open("data/DD-IPI/R2/def_strategy_counter.pkl", "rb")
    def_strat_count = pickle.load(the_file)
    the_file.close()

    plt.figure(figsize=(figure_width, figure_high + 1))
    max_length = 0
    for key in def_strat_count.keys():
        if max_length < len(def_strat_count[key]):
            max_length = len(def_strat_count[key])

    average_def_strat = []
    for index in range(max_length):
        sum_on_index = 0
        number_on_index = 0
        def_strat = np.zeros(8)
        for key in def_strat_count.keys():
            if len(def_strat_count[key]) > 0:
                def_strat[def_strat_count[key][0]] += 1
                def_strat_count[key].pop(0)
                number_on_index += 1
        average_def_strat.append((def_strat / number_on_index).tolist())
    average_def_strat = np.array(average_def_strat)

    all_linestyle = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    for index in range(8):
        plt.plot(range(max_length),
                 average_def_strat[:, index],
                 linestyle=all_linestyle[index],
                 label=f"Stra {index + 1}")

    plt.legend(prop={"size": legend_size},
               ncol=4,
               bbox_to_anchor=(0, 1, 1, 0),
               loc='lower left',
               mode="expand")

    plt.xlabel("number of games", fontsize=font_size)
    plt.ylabel("$P_S^D$", fontsize=font_size)
    plt.xticks(fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    plt.tight_layout()
    # plt.savefig("Figure/DD-IPI-def-Strat.eps", dpi=figure_dpi)
    plt.show()

# Attacker Strategy DD-PI
def DD_PI_att_strat_count():
    the_file = open("data/DD-PI/R2/att_strategy_counter.pkl", "rb")
    att_strat_count = pickle.load(the_file)
    the_file.close()

    plt.figure(figsize=(figure_width, figure_high + 1))
    max_length = 0
    for key in att_strat_count.keys():
        if max_length < len(att_strat_count[key]):
            max_length = len(att_strat_count[key])

    average_att_strat = []
    for index in range(max_length):
        sum_on_index = 0
        number_on_index = 0
        att_strat = np.zeros(8)
        for key in att_strat_count.keys():
            if len(att_strat_count[key]) > 0:
                att_strat[att_strat_count[key][0]] += 1
                att_strat_count[key].pop(0)
                number_on_index += 1
        average_att_strat.append((att_strat / number_on_index).tolist())
    average_att_strat = np.array(average_att_strat)

    for index in range(8):
        plt.plot(range(max_length), average_att_strat[:, index], linestyle=all_linestyle[index],
                 label=f"Stra {index + 1}")

    plt.legend(prop={"size": legend_size},
               ncol=4,
               bbox_to_anchor=(0, 1, 1, 0),
               loc='lower left',
               mode="expand")

    plt.xlabel("number of games", fontsize=font_size)
    plt.ylabel("$P_S^A$", fontsize=font_size)
    plt.xticks(fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    plt.tight_layout()
    # plt.savefig("Figure/DD-PI-att-Strat.eps", dpi=figure_dpi)
    plt.show()

# Defender Strategy DD-PI
def DD_PI_def_strat_count():
    the_file = open("data/DD-PI/R2/def_strategy_counter.pkl", "rb")
    def_strat_count = pickle.load(the_file)
    the_file.close()

    plt.figure(figsize=(figure_width, figure_high + 1))
    max_length = 0
    for key in def_strat_count.keys():
        if max_length < len(def_strat_count[key]):
            max_length = len(def_strat_count[key])

    average_def_strat = []
    for index in range(max_length):
        sum_on_index = 0
        number_on_index = 0
        def_strat = np.zeros(8)
        for key in def_strat_count.keys():
            if len(def_strat_count[key]) > 0:
                def_strat[def_strat_count[key][0]] += 1
                def_strat_count[key].pop(0)
                number_on_index += 1
        average_def_strat.append((def_strat / number_on_index).tolist())
    average_def_strat = np.array(average_def_strat)

    all_linestyle = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    for index in range(8):
        plt.plot(range(max_length), average_def_strat[:, index], linestyle=all_linestyle[index],
                 label=f"Stra {index + 1}")

    plt.legend(prop={"size": legend_size},
               ncol=4,
               bbox_to_anchor=(0, 1, 1, 0),
               loc='lower left',
               mode="expand")

    plt.xlabel("number of games", fontsize=font_size)
    plt.ylabel("$P_S^D$", fontsize=font_size)
    plt.xticks(fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    plt.tight_layout()
    plt.savefig("Figure/DD-PI-def-Strat.eps", dpi=figure_dpi)


# Attacker Strategy No-DD-IPI
def No_DD_IPI_att_strat_count():
    the_file = open("data/No-DD-IPI/R2/att_strategy_counter.pkl", "rb")
    att_strat_count = pickle.load(the_file)
    the_file.close()

    plt.figure(figsize=(figure_width, figure_high + 1))
    max_length = 0
    for key in att_strat_count.keys():
        if max_length < len(att_strat_count[key]):
            max_length = len(att_strat_count[key])

    average_att_strat = []
    for index in range(max_length):
        sum_on_index = 0
        number_on_index = 0
        att_strat = np.zeros(8)
        for key in att_strat_count.keys():
            if len(att_strat_count[key]) > 0:
                att_strat[att_strat_count[key][0]] += 1
                att_strat_count[key].pop(0)
                number_on_index += 1
        average_att_strat.append((att_strat / number_on_index).tolist())
    average_att_strat = np.array(average_att_strat)

    for index in range(8):
        plt.plot(range(max_length), average_att_strat[:, index], linestyle=all_linestyle[index],
                 label=f"Stra {index + 1}")

    plt.legend(prop={"size": legend_size},
               ncol=4,
               bbox_to_anchor=(0, 1, 1, 0),
               loc='lower left',
               mode="expand")
    plt.xlabel("number of games", fontsize=font_size)
    plt.ylabel("$P_S^A$", fontsize=font_size)
    plt.xticks(fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    plt.tight_layout()
    plt.savefig("Figure/No-DD-IPI-att-Strat.eps", dpi=figure_dpi)


# Defender Strategy No-DD-IPI
def No_DD_IPI_def_strat_count():
    the_file = open("data/No-DD-IPI/R2/def_strategy_counter.pkl", "rb")
    def_strat_count = pickle.load(the_file)
    the_file.close()

    plt.figure(figsize=(figure_width, figure_high + 1))
    max_length = 0
    for key in def_strat_count.keys():
        if max_length < len(def_strat_count[key]):
            max_length = len(def_strat_count[key])

    average_def_strat = []
    for index in range(max_length):
        sum_on_index = 0
        number_on_index = 0
        def_strat = np.zeros(8)
        for key in def_strat_count.keys():
            if len(def_strat_count[key]) > 0:
                def_strat[def_strat_count[key][0]] += 1
                def_strat_count[key].pop(0)
                number_on_index += 1
        average_def_strat.append((def_strat / number_on_index).tolist())
    average_def_strat = np.array(average_def_strat)

    all_linestyle = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    for index in range(8):
        plt.plot(range(max_length), average_def_strat[:, index], linestyle=all_linestyle[index],
                 label=f"Stra {index + 1}")

    plt.legend(prop={"size": legend_size},
               ncol=4,
               bbox_to_anchor=(0, 1, 1, 0),
               loc='lower left',
               mode="expand")
    plt.xlabel("number of games", fontsize=font_size)
    plt.ylabel("$P_S^D$", fontsize=font_size)
    plt.xticks(fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    plt.tight_layout()
    plt.savefig("Figure/No-DD-IPI-def-Strat.eps", dpi=figure_dpi)


# Attacker Strategy No-DD-PI
def No_DD_PI_att_strat_count():
    the_file = open("data/No-DD-PI/R2/att_strategy_counter.pkl", "rb")
    att_strat_count = pickle.load(the_file)
    the_file.close()

    plt.figure(figsize=(figure_width, figure_high + 1))
    max_length = 0
    for key in att_strat_count.keys():
        if max_length < len(att_strat_count[key]):
            max_length = len(att_strat_count[key])

    average_att_strat = []
    for index in range(max_length):
        sum_on_index = 0
        number_on_index = 0
        att_strat = np.zeros(8)
        for key in att_strat_count.keys():
            if len(att_strat_count[key]) > 0:
                att_strat[att_strat_count[key][0]] += 1
                att_strat_count[key].pop(0)
                number_on_index += 1
        average_att_strat.append((att_strat / number_on_index).tolist())
    average_att_strat = np.array(average_att_strat)

    for index in range(8):
        plt.plot(range(max_length), average_att_strat[:, index], linestyle=all_linestyle[index],
                 label=f"Stra {index + 1}")

    plt.legend(prop={"size": legend_size},
               ncol=4,
               bbox_to_anchor=(0, 1, 1, 0),
               loc='lower left',
               mode="expand")
    plt.xlabel("number of games", fontsize=font_size)
    plt.ylabel("$P_S^A$", fontsize=font_size)
    plt.xticks(fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    plt.tight_layout()
    # plt.savefig("Figure/No-DD-PI-att-Strat.eps", dpi=figure_dpi)
    plt.show()


# Defender Strategy No-DD-PI
def No_DD_PI_def_strat_count():
    the_file = open("data/No-DD-PI/R2/def_strategy_counter.pkl", "rb")
    def_strat_count = pickle.load(the_file)
    the_file.close()

    plt.figure(figsize=(figure_width, figure_high + 1))
    max_length = 0
    for key in def_strat_count.keys():
        if max_length < len(def_strat_count[key]):
            max_length = len(def_strat_count[key])

    average_def_strat = []
    for index in range(max_length):
        sum_on_index = 0
        number_on_index = 0
        def_strat = np.zeros(8)
        for key in def_strat_count.keys():
            if len(def_strat_count[key]) > 0:
                def_strat[def_strat_count[key][0]] += 1
                def_strat_count[key].pop(0)
                number_on_index += 1
        average_def_strat.append((def_strat / number_on_index).tolist())
    average_def_strat = np.array(average_def_strat)

    all_linestyle = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    for index in range(8):
        plt.plot(range(max_length), average_def_strat[:, index], linestyle=all_linestyle[index],
                 label=f"Stra {index + 1}")

    plt.legend(prop={"size": legend_size},
               ncol=4,
               bbox_to_anchor=(0, 1, 1, 0),
               loc='lower left',
               mode="expand")
    plt.xlabel("number of games", fontsize=font_size)
    plt.ylabel("$P_S^D$", fontsize=font_size)
    plt.xticks(fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    plt.tight_layout()
    plt.savefig("Figure/No-DD-PI-def-Strat.eps", dpi=figure_dpi)


# attacker uncertainty average result
def display_att_uncertain(schemes):
    plt.figure(figsize=(figure_width, figure_high))
    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/R3/attacker_uncertainty.pkl", "rb")
        att_uncertainty_history = pickle.load(the_file)
        the_file.close()

        max_length = 0
        max_index = 0
        for key in att_uncertainty_history.keys():
            if max_length < len(att_uncertainty_history[key]):
                max_length = len(att_uncertainty_history[key])
                max_index = key

        # print(att_uncertainty_history[max_index])
        # plt.plot(range(len(att_uncertainty_history[max_index])), att_uncertainty_history[max_index])

        average_att_uncertainty = []
        for index in range(max_length):
            sum_on_index = 0
            number_on_index = 0
            for key in att_uncertainty_history.keys():
                if len(att_uncertainty_history[key]) > 0:
                    sum_on_index += att_uncertainty_history[key][0]
                    att_uncertainty_history[key].pop(0)
                    number_on_index += 1
            average_att_uncertainty.append(sum_on_index / number_on_index)

        x_values = range(len(average_att_uncertainty))
        y_values = average_att_uncertainty
        plt.plot(x_values, y_values, linestyle=all_linestyle[schemes_index], label=schemes[schemes_index],
                 linewidth=figure_linewidth, marker=marker_list[schemes_index], markevery=50, markersize=marker_size)
    # plt.legend(prop={"size":legend_size}, ncol=2, bbox_to_anchor=(0, 1, 1, 0),
    #               loc='lower left', fontsize='large',mode="expand")
    # plt.legend(prop={"size":legend_size}, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.xlabel("number of games", fontsize=font_size)
    plt.ylabel("Uncertainty", fontsize=font_size)
    plt.xticks(fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    plt.tight_layout()
    plt.savefig("Figure/att-uncertain-NG.eps", dpi=figure_dpi)


# defender uncertainty average result
def display_def_uncertain(schemes):
    plt.figure(figsize=(figure_width, figure_high))
    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/R3/defender_uncertainty.pkl", "rb")
        def_uncertainty_history = pickle.load(the_file)
        the_file.close()

        max_length = 0
        for key in def_uncertainty_history.keys():
            if max_length < len(def_uncertainty_history[key]):
                max_length = len(def_uncertainty_history[key])

        average_def_uncertainty = []
        for index in range(max_length):
            sum_on_index = 0
            number_on_index = 0
            for key in def_uncertainty_history.keys():
                if len(def_uncertainty_history[key]) > 0:
                    sum_on_index += def_uncertainty_history[key][0]
                    def_uncertainty_history[key].pop(0)
                    number_on_index += 1
            average_def_uncertainty.append(sum_on_index / number_on_index)

        x_values = range(len(average_def_uncertainty))
        y_values = average_def_uncertainty
        plt.plot(x_values, y_values, linestyle=all_linestyle[schemes_index], label=schemes[schemes_index],
                 linewidth=figure_linewidth, marker=marker_list[schemes_index], markevery=50, markersize=marker_size)
    # plt.legend(prop={"size":legend_size}, ncol=2, bbox_to_anchor=(0, 1, 1, 0),
    #               loc='lower left', fontsize='large',mode="expand")
    plt.xlabel("number of games", fontsize=font_size)
    plt.ylabel("Uncertainty", fontsize=font_size)
    plt.xticks(fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    plt.tight_layout()
    plt.savefig("Figure/def-uncertain-NG.eps", dpi=figure_dpi)


# IDS FPR
def display_FPR(schemes):
    plt.figure(figsize=(figure_width, figure_high+ 0.75))
    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/R4/FPR.pkl", "rb")
        FPR_history = pickle.load(the_file)
        the_file.close()

        max_length = 0
        for key in FPR_history.keys():
            if max_length < len(FPR_history[key]):
                max_length = len(FPR_history[key])

        average_FPR = []
        for index in range(max_length):
            sum_on_index = 0
            number_on_index = 0
            for key in FPR_history.keys():
                if len(FPR_history[key]) > 0:
                    sum_on_index += FPR_history[key][0]
                    FPR_history[key].pop(0)
                    number_on_index += 1
            average_FPR.append(sum_on_index / number_on_index)

        x_values = range(len(average_FPR))
        y_values = average_FPR

        plt.plot(x_values, y_values, linestyle=all_linestyle[schemes_index], label=schemes[schemes_index],
                 linewidth=figure_linewidth, marker=marker_list[schemes_index], markevery=50, markersize=marker_size)
    plt.legend(prop={"size": legend_size},
               ncol=4,
               bbox_to_anchor=(-0.13, 1, 1.15, 0),
               loc='lower left',
               mode="expand")
    plt.xlabel("number of games", fontsize=font_size)
    plt.ylabel("FPR", fontsize=font_size)
    plt.xticks(fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    plt.tight_layout()
    os.makedirs("Figure/TPR_or_FPR", exist_ok=True)
    os.makedirs("Figure (PNG)/TPR_or_FPR", exist_ok=True)
    plt.savefig("Figure/TPR_or_FPR/FPR-NG.eps", dpi=figure_dpi)
    plt.savefig("Figure (PNG)/TPR_or_FPR/FPR-NG.png", dpi=figure_dpi)
    plt.show()


# IDS TPR
def display_TPR(schemes):
    plt.figure(figsize=(figure_width, figure_high + 0.75))
    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/R4/TPR.pkl", "rb")
        TPR_history = pickle.load(the_file)
        the_file.close()

        max_length = 0
        for key in TPR_history.keys():
            if max_length < len(TPR_history[key]):
                max_length = len(TPR_history[key])

        average_TPR = []
        for index in range(max_length):
            sum_on_index = 0
            number_on_index = 0
            for key in TPR_history.keys():
                if len(TPR_history[key]) > 0:
                    sum_on_index += TPR_history[key][0]
                    TPR_history[key].pop(0)
                    number_on_index += 1
            average_TPR.append(sum_on_index / number_on_index)

        x_values = range(len(average_TPR))
        y_values = average_TPR

        plt.plot(x_values, y_values, linestyle=all_linestyle[schemes_index], label=schemes[schemes_index],
                 linewidth=figure_linewidth, marker=marker_list[schemes_index], markevery=50, markersize=marker_size)
    plt.legend(prop={"size": legend_size},
               ncol=4,
               bbox_to_anchor=(-0.13, 1, 1.15, 0),
               loc='lower left',
               mode="expand")
    plt.xlabel("number of games", fontsize=font_size)
    plt.ylabel("TPR", fontsize=font_size)
    plt.xticks(fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    plt.tight_layout()
    os.makedirs("Figure/TPR_or_FPR", exist_ok=True)
    os.makedirs("Figure (PNG)/TPR_or_FPR", exist_ok=True)
    plt.savefig("Figure/TPR_or_FPR/TPR-NG.eps", dpi=figure_dpi)
    plt.savefig("Figure (PNG)/TPR_or_FPR/TPR-NG.png", dpi=figure_dpi)
    plt.show()


# ROC
def display_TPR_2(schemes):
    plt.figure(figsize=(figure_width, figure_high))
    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/R4/TPR.pkl", "rb")
        TPR_history = pickle.load(the_file)
        the_file.close()
        the_file = open("data/" + schemes[schemes_index] + "/R4/FPR.pkl", "rb")
        FPR_history = pickle.load(the_file)
        the_file.close()

        max_length = 0
        for key in TPR_history.keys():
            if max_length < len(TPR_history[key]):
                max_length = len(TPR_history[key])

        average_TPR = []
        for index in range(max_length):
            sum_on_index = 0
            number_on_index = 0
            for key in TPR_history.keys():
                if len(TPR_history[key]) > 0:
                    sum_on_index += TPR_history[key][0]
                    TPR_history[key].pop(0)
                    number_on_index += 1
            average_TPR.append(sum_on_index / number_on_index)

        average_FPR = []
        for index in range(max_length):
            sum_on_index = 0
            number_on_index = 0
            for key in FPR_history.keys():
                if len(FPR_history[key]) > 0:
                    sum_on_index += FPR_history[key][0]
                    FPR_history[key].pop(0)
                    number_on_index += 1
            average_FPR.append(sum_on_index / number_on_index)

        x_values = average_FPR

        #     print(average_TPR)
        #     print(average_FPR)
        y_values = average_TPR

        plt.plot(x_values, y_values, linestyle=all_linestyle[schemes_index], label=schemes[schemes_index])
    plt.legend(prop={"size": legend_size})
    plt.xlabel("FPR", fontsize=font_size)
    plt.ylabel("TPR", fontsize=font_size)
    plt.xticks(fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    plt.tight_layout()
    plt.savefig("Figure/ROC-NG.eps", dpi=figure_dpi)


# Time to System Fail
def display_TTSF(schemes):
    granularity = 1

    plt.figure(figsize=(figure_width, figure_high))
    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/Time_to_SF.pkl", "rb")
        Time_to_SF = pickle.load(the_file)
        the_file.close()

        #     print(max(Time_to_SF.values()))
        x_scales = np.zeros(max(Time_to_SF.values()) + 1)
        y_scales = np.zeros(max(Time_to_SF.values()) + 1)
        for value in Time_to_SF.values():
            x_scales[int(value / granularity) * granularity] = int(value / granularity) * granularity + 1
            y_scales[int(value / granularity) * granularity] += 1

        # pop zero in array
        x_scales = x_scales[x_scales != 0]
        y_scales = y_scales[y_scales != 0]
        #     plt.scatter(x_scales, y_scales, label=schemes[schemes_index])
        plt.plot(x_scales, y_scales, label=schemes[schemes_index])

    plt.legend(prop={"size": legend_size})
    plt.xlabel(f"Time to System Failure (granularity={granularity})", fontsize=font_size)
    plt.ylabel("Frequency of Simulation", fontsize=font_size)
    plt.xticks(fontsize=axis_size)
    plt.yticks(fontsize=axis_size)


# MTTSF
def display_MTTSF(schemes):
    plt.figure(figsize=(figure_width, figure_high))
    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/VUB/MTTSF.pkl", "rb")
        MTTSF = pickle.load(the_file)
        the_file.close()

        x_scales_all_change = ["2", "4", "6", "8", "10"]

        plt.plot(x_scales_all_change,
                 MTTSF,
                 linestyle=all_linestyle[schemes_index],
                 label=schemes[schemes_index],
                 linewidth=figure_linewidth,
                 marker=marker_list[schemes_index], markersize=marker_size)

    # plt.legend(prop={"size":legend_size}, ncol=2, bbox_to_anchor=(0, 1, 1, 0),
    #               loc='lower left', fontsize='large',mode="expand")
    plt.xlabel("vulnerability upper bound", fontsize=font_size)
    plt.ylabel("MTTSF", fontsize=font_size)
    plt.xticks(fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    plt.tight_layout()
    plt.savefig("Figure/VUB/MTTSF-VV.eps", dpi=figure_dpi)
    plt.savefig("Figure (PNG)/VUB/MTTSF-VV.png", dpi=figure_dpi)
    plt.show()


# R8 attacker cost in changing Vul
def varying_vul_att_cost(schemes):
    plt.figure(figsize=(figure_width, figure_high))
    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/VUB/att_cost.pkl",
                        "rb")
        att_cost_all_result = pickle.load(the_file)
        the_file.close()

        #     x_scales = [
        #         f"IoT:(1,{vul_range[1][0]})\nWeb&Data:(1,{vul_range[0][0]})",
        #         f"IoT:(1,{vul_range[1][1]})\nWeb&Data:(1,{vul_range[0][1]})",
        #         f"IoT:(1,{vul_range[1][2]})\nWeb&Data:(1,{vul_range[0][2]})",
        #         f"IoT:(1,{vul_range[1][3]})\nWeb&Data:(1,{vul_range[0][3]})",
        #         f"IoT:(1,{vul_range[1][4]})\nWeb&Data:(1,{vul_range[0][4]})"
        #     ]
        x_scales = ["2", "4", "6", "8", "10"]

        plt.plot(x_scales,
                 att_cost_all_result,
                 linestyle=all_linestyle[schemes_index],
                 label=schemes[schemes_index],
                 linewidth=figure_linewidth,
                 marker=marker_list[schemes_index], markersize=marker_size)

    # plt.legend(prop={"size":legend_size}, ncol=2, bbox_to_anchor=(0, 1, 1, 0),
    #               loc='lower left', fontsize='large',mode="expand")
    plt.xlabel("vulnerability upper bound", fontsize=font_size)
    plt.ylabel("Attack Cost", fontsize=font_size)
    plt.xticks(fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    plt.tight_layout()
    plt.savefig("Figure/VUB/att-cost-VV.eps", dpi=figure_dpi)
    plt.savefig("Figure (PNG)/VUB/att-cost-VV.png", dpi=figure_dpi)
    plt.show()


# R8 defender cost in changing Vul
def varying_vul_def_cost(schemes):
    plt.figure(figsize=(figure_width, figure_high))
    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/VUB/def_cost.pkl",
                        "rb")
        def_cost_all_result = pickle.load(the_file)
        the_file.close()

        #     x_scales = [
        #         f"IoT:(1,{vul_range[1][0]})\nWeb&Data:(1,{vul_range[0][0]})",
        #         f"IoT:(1,{vul_range[1][1]})\nWeb&Data:(1,{vul_range[0][1]})",
        #         f"IoT:(1,{vul_range[1][2]})\nWeb&Data:(1,{vul_range[0][2]})",
        #         f"IoT:(1,{vul_range[1][3]})\nWeb&Data:(1,{vul_range[0][3]})",
        #         f"IoT:(1,{vul_range[1][4]})\nWeb&Data:(1,{vul_range[0][4]})"
        #     ]

        x_scales = ["2", "4", "6", "8", "10"]

        plt.plot(x_scales,
                 def_cost_all_result,
                 linestyle=all_linestyle[schemes_index],
                 label=schemes[schemes_index],
                 linewidth=figure_linewidth,
                 marker=marker_list[schemes_index], markersize=marker_size)

    # plt.legend(prop={"size":legend_size}, ncol=2, bbox_to_anchor=(0, 1, 1, 0),
    #               loc='lower left', fontsize='large',mode="expand")
    plt.xlabel("vulnerability upper bound", fontsize=font_size)
    plt.ylabel("Defense Cost", fontsize=font_size)
    plt.xticks(fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    plt.tight_layout()
    plt.savefig("Figure/VUB/def-cost-VV.eps", dpi=figure_dpi)
    plt.savefig("Figure (PNG)/VUB/def-cost-VV.png", dpi=figure_dpi)
    plt.show()


# R9 attacker HEU in changing Vul
def varying_vul_att_HEU(schemes):
    plt.figure(figsize=(figure_width, figure_high))
    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/VUB/att_HEU.pkl", "rb")
        att_HEU_all_result = pickle.load(the_file)
        the_file.close()

        #     x_scales = [
        #         f"IoT:(1,{vul_range[1][0]})\nWeb&Data:(1,{vul_range[0][0]})",
        #         f"IoT:(1,{vul_range[1][1]})\nWeb&Data:(1,{vul_range[0][1]})",
        #         f"IoT:(1,{vul_range[1][2]})\nWeb&Data:(1,{vul_range[0][2]})",
        #         f"IoT:(1,{vul_range[1][3]})\nWeb&Data:(1,{vul_range[0][3]})",
        #         f"IoT:(1,{vul_range[1][4]})\nWeb&Data:(1,{vul_range[0][4]})"
        #     ]

        x_scales = ["2", "4", "6", "8", "10"]

        plt.plot(x_scales,
                 att_HEU_all_result,
                 linestyle=all_linestyle[schemes_index],
                 label=schemes[schemes_index],
                 linewidth=figure_linewidth,
                 marker=marker_list[schemes_index], markersize=marker_size)

    # plt.legend(prop={"size":legend_size}, ncol=2, bbox_to_anchor=(0, 1, 1, 0),
    #               loc='lower left', fontsize='large',mode="expand")
    plt.xlabel("vulnerability upper bound", fontsize=font_size)
    plt.ylabel("AHEU", fontsize=font_size)
    plt.xticks(fontsize=axis_size)
    plt.yticks(fontsize=axis_size);
    plt.tight_layout()
    plt.savefig("Figure/VUB/att-HEU-VV.eps", dpi=figure_dpi)
    plt.savefig("Figure (PNG)/VUB/att-HEU-VV.png", dpi=figure_dpi)
    plt.show()


# R9 defender HEU in changing Vul
def varying_vul_def_HEU(schemes):
    plt.figure(figsize=(figure_width, figure_high))
    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/VUB/def_HEU.pkl", "rb")
        def_HEU_all_result = pickle.load(the_file)
        the_file.close()

        #     x_scales = [
        #         f"IoT:(1,{vul_range[1][0]})\nWeb&Data:(1,{vul_range[0][0]})",
        #         f"IoT:(1,{vul_range[1][1]})\nWeb&Data:(1,{vul_range[0][1]})",
        #         f"IoT:(1,{vul_range[1][2]})\nWeb&Data:(1,{vul_range[0][2]})",
        #         f"IoT:(1,{vul_range[1][3]})\nWeb&Data:(1,{vul_range[0][3]})",
        #         f"IoT:(1,{vul_range[1][4]})\nWeb&Data:(1,{vul_range[0][4]})"
        #     ]

        x_scales = ["2", "4", "6", "8", "10"]

        plt.plot(x_scales,
                 def_HEU_all_result,
                 linestyle=all_linestyle[schemes_index],
                 label=schemes[schemes_index],
                 linewidth=figure_linewidth,
                 marker=marker_list[schemes_index], markersize=marker_size)

    # plt.legend(prop={"size":legend_size}, ncol=2, bbox_to_anchor=(0, 1, 1, 0),
    #               loc='lower left', fontsize='large',mode="expand")
    plt.xlabel("vulnerability upper bound", fontsize=font_size)
    plt.ylabel("DHEU", fontsize=font_size)
    plt.xticks(fontsize=axis_size);
    plt.yticks(fontsize=axis_size);
    plt.tight_layout()
    plt.savefig("Figure/VUB/def-HEU-VV.eps", dpi=figure_dpi)
    plt.savefig("Figure (PNG)/VUB/def-HEU-VV.png", dpi=figure_dpi)
    plt.show()


# R10 attacker Uncertainty in changing Vul
def varying_vul_att_uncertain(schemes):
    plt.figure(figsize=(figure_width, figure_high))
    for schemes_index in range(len(schemes)):
        the_file = open(
            "data/" + schemes[schemes_index] + "/VUB/att_uncertainty.pkl", "rb")
        att_uncertainty_all_result = pickle.load(the_file)
        the_file.close()

        #     x_scales = [
        #         f"IoT:(1,{vul_range[1][0]})\nWeb&Data:(1,{vul_range[0][0]})",
        #         f"IoT:(1,{vul_range[1][1]})\nWeb&Data:(1,{vul_range[0][1]})",
        #         f"IoT:(1,{vul_range[1][2]})\nWeb&Data:(1,{vul_range[0][2]})",
        #         f"IoT:(1,{vul_range[1][3]})\nWeb&Data:(1,{vul_range[0][3]})",
        #         f"IoT:(1,{vul_range[1][4]})\nWeb&Data:(1,{vul_range[0][4]})"
        #     ]

        x_scales = ["2", "4", "6", "8", "10"]

        plt.plot(x_scales,
                 att_uncertainty_all_result,
                 linestyle=all_linestyle[schemes_index],
                 label=schemes[schemes_index],
                 linewidth=figure_linewidth,
                 marker=marker_list[schemes_index], markersize=marker_size)

    # plt.legend(prop={"size":15}, ncol=4, bbox_to_anchor=(0, 1, 1, 0),
    #               loc='lower left', fontsize='large',mode="expand")
    plt.xlabel("vulnerability upper bound", fontsize=font_size)
    plt.ylabel("Uncertaity", fontsize=font_size)
    plt.xticks(fontsize=axis_size);
    plt.yticks(fontsize=axis_size);
    plt.tight_layout()
    plt.savefig("Figure/VUB/att-uncertain-VV.eps", dpi=figure_dpi)
    plt.savefig("Figure (PNG)/VUB/att-uncertain-VV.png", dpi=figure_dpi)
    plt.show()


# R10 defender Uncertainty in changing Vul
def varying_vul_def_uncertain(schemes):
    plt.figure(figsize=(figure_width, figure_high))
    for schemes_index in range(len(schemes)):
        the_file = open(
            "data/" + schemes[schemes_index] + "/VUB/def_uncertainty.pkl", "rb")
        def_uncertainty_all_result = pickle.load(the_file)
        the_file.close()

        #     x_scales = [
        #         f"IoT:(1,{vul_range[1][0]})\nWeb&Data:(1,{vul_range[0][0]})",
        #         f"IoT:(1,{vul_range[1][1]})\nWeb&Data:(1,{vul_range[0][1]})",
        #         f"IoT:(1,{vul_range[1][2]})\nWeb&Data:(1,{vul_range[0][2]})",
        #         f"IoT:(1,{vul_range[1][3]})\nWeb&Data:(1,{vul_range[0][3]})",
        #         f"IoT:(1,{vul_range[1][4]})\nWeb&Data:(1,{vul_range[0][4]})"
        #     ]

        x_scales = ["2", "4", "6", "8", "10"]

        plt.plot(x_scales,
                 def_uncertainty_all_result,
                 linestyle=all_linestyle[schemes_index],
                 label=schemes[schemes_index],
                 linewidth=figure_linewidth,
                 marker=marker_list[schemes_index], markersize=marker_size)

    # plt.legend(prop={"size":legend_size}, ncol=2, bbox_to_anchor=(0, 1, 1, 0),
    #               loc='lower left', fontsize='large',mode="expand")
    plt.xlabel("vulnerability upper bound", fontsize=font_size)
    plt.ylabel("Uncertaity", fontsize=font_size)
    plt.xticks(fontsize=axis_size);
    plt.yticks(fontsize=axis_size);
    plt.tight_layout()
    plt.savefig("Figure/VUB/def-uncertain-VV.eps", dpi=figure_dpi)
    plt.savefig("Figure (PNG)/VUB/def-uncertain-VV.png", dpi=figure_dpi)
    plt.show()


# R11 FPR in changing Vul
def varying_vul_FPR(schemes):
    plt.figure(figsize=(figure_width, figure_high))
    for schemes_index in range(len(schemes)):
        the_file = open(
            "data/" + schemes[schemes_index] + "/VUB/FPR.pkl", "rb")
        FPR_all_result = pickle.load(the_file)
        the_file.close()

        #     x_scales = [
        #         f"IoT:(1,{vul_range[1][0]})\nWeb&Data:(1,{vul_range[0][0]})",
        #         f"IoT:(1,{vul_range[1][1]})\nWeb&Data:(1,{vul_range[0][1]})",
        #         f"IoT:(1,{vul_range[1][2]})\nWeb&Data:(1,{vul_range[0][2]})",
        #         f"IoT:(1,{vul_range[1][3]})\nWeb&Data:(1,{vul_range[0][3]})",
        #         f"IoT:(1,{vul_range[1][4]})\nWeb&Data:(1,{vul_range[0][4]})"
        #     ]

        x_scales = ["2", "4", "6", "8", "10"]

        plt.plot(x_scales,
                 FPR_all_result,
                 linestyle=all_linestyle[schemes_index],
                 label=schemes[schemes_index],
                 linewidth=figure_linewidth,
                 marker=marker_list[schemes_index], markersize=marker_size)

    # plt.legend(prop={"size": legend_size}, ncol=2, bbox_to_anchor=(0, 1, 1, 0),
    #            loc='lower left', fontsize='large', mode="expand")
    plt.xlabel("vulnerability upper bound", fontsize=font_size)
    plt.ylabel("FPR", fontsize=font_size)
    plt.xticks(fontsize=axis_size);
    plt.yticks(fontsize=axis_size);
    plt.tight_layout()
    plt.savefig("Figure/VUB/FPR-VV.eps", dpi=figure_dpi)
    plt.savefig("Figure (PNG)/VUB/FPR-VV.png", dpi=figure_dpi)
    plt.show()


# R11 TPR in changing Vul
def varying_vul_TPR(schemes):
    plt.figure(figsize=(figure_width, figure_high))
    for schemes_index in range(len(schemes)):
        the_file = open(
            "data/" + schemes[schemes_index] + "/VUB/TPR.pkl", "rb")
        TPR_all_result = pickle.load(the_file)
        the_file.close()

        #     x_scales = [
        #         f"IoT:(1,{vul_range[1][0]})\nWeb&Data:(1,{vul_range[0][0]})",
        #         f"IoT:(1,{vul_range[1][1]})\nWeb&Data:(1,{vul_range[0][1]})",
        #         f"IoT:(1,{vul_range[1][2]})\nWeb&Data:(1,{vul_range[0][2]})",
        #         f"IoT:(1,{vul_range[1][3]})\nWeb&Data:(1,{vul_range[0][3]})",
        #         f"IoT:(1,{vul_range[1][4]})\nWeb&Data:(1,{vul_range[0][4]})"
        #     ]

        x_scales = ["2", "4", "6", "8", "10"]

        plt.plot(x_scales,
                 TPR_all_result,
                 linestyle=all_linestyle[schemes_index],
                 label=schemes[schemes_index],
                 linewidth=figure_linewidth,
                 marker=marker_list[schemes_index], markersize=marker_size)

    # plt.legend(prop={"size":legend_size}, ncol=2, bbox_to_anchor=(0, 1, 1, 0),
    #               loc='lower left', fontsize='large',mode="expand")
    plt.xlabel("vulnerability upper bound", fontsize=font_size)
    plt.ylabel("TPR", fontsize=font_size)
    plt.xticks(fontsize=axis_size);
    plt.yticks(fontsize=axis_size);
    plt.tight_layout()
    plt.savefig("Figure/VUB/TPR-VV.eps", dpi=figure_dpi)
    plt.savefig("Figure (PNG)/VUB/TPR-VV.png", dpi=figure_dpi)
    plt.show()


# criticality distribution
def display_criticality(schemes):
    plt.figure(figsize=(figure_width, figure_high))
    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/R_self_1/criticality.pkl", "rb")
        criti_history = pickle.load(the_file)
        the_file.close()

        total_criticality = np.zeros(100000)
        for key in criti_history.keys():
            total_criticality += criti_history[key]
        #         print(total_criticality)

        last_ele = np.max(np.nonzero(total_criticality))
        total_criticality = total_criticality[:last_ele + 1]
        #     total_criticality = total_criticality/len(criti_history)
        #     print(sum(total_criticality))

        x_values = np.array(range(len(total_criticality))) / 10000
        y_values = total_criticality / sum(total_criticality)

        plt.plot(x_values, y_values, linestyle=all_linestyle[schemes_index], label=schemes[schemes_index])
        plt.yscale('log')

    plt.legend(prop={"size": legend_size})
    plt.xlabel("Range of Criticality", fontsize=font_size)
    plt.ylabel("Frequency in Log_10", fontsize=font_size)
    plt.xticks(fontsize=axis_size);
    plt.yticks(fontsize=axis_size);


# # Evict attacker reason
# schemes = ["DD-IPI", "DD-PI", "No-DD-IPI", "No-DD-PI"]

# for schemes_index in range(len(schemes)):
#     the_file = open("data/"+schemes[schemes_index]+"/R_self_2/evict_reason.pkl", "rb")
#     evict_reson_history = pickle.load(the_file)
#     the_file.close()

#     plt.figure(figsize=(figure_width, figure_high))
#     total_reson = np.zeros(2)
#     for key in evict_reson_history:
#         total_reson += evict_reson_history[key]

#     plt.bar(["Be outside", "Honeypot"], total_reson)

#     plt.title(schemes[schemes_index], fontsize=font_size)
#     plt.xlabel("Reasons for Creating New Attacker", fontsize=font_size)
#     plt.ylabel("number of attacker Created", fontsize=font_size)




# # System Failure reason
# schemes = ["DD-IPI", "DD-PI", "No-DD-IPI", "No-DD-PI"]

# for schemes_index in range(len(schemes)):
#     the_file = open("data/"+schemes[schemes_index]+"/R_self_3/system_fail.pkl", "rb")
#     SysFail_reason = pickle.load(the_file)
#     the_file.close()

#     plt.figure(figsize=(figure_width, figure_high))
#     print(SysFail_reason)
#     x_values = ["Att Strategy 8\nSuccess","Total Compromised Criticality\nLarger Than ", "Too Many Nodes\nEvicted"]
#     y_values = SysFail_reason
#     plt.bar(x_values, y_values)

#     plt.title(schemes[schemes_index], fontsize=font_size)
#     plt.xlabel("Reasons for System Failure", fontsize=font_size)
#     plt.ylabel("number of simulation", fontsize=font_size)


def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    decimal_number = 3
    for rect in rects:
        height = round(rect.get_height(), decimal_number)
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


# Evict attacker reason
def display_eviction(schemes):
    fig, ax = plt.subplots(figsize=(figure_width, figure_high))
    width = 0.2
    shift_value = [- width / 2 - width, - width / 2, + width / 2, width / 2 + width]
    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/R_self_2/evict_reason.pkl", "rb")
        evict_reson_history = pickle.load(the_file)
        the_file.close()

        total_reson = np.zeros(2)
        for key in evict_reson_history:
            total_reson += evict_reson_history[key]

        y_values = total_reson
        temp_x = np.arange(len(y_values))
        rects = ax.bar(temp_x + shift_value[schemes_index], y_values, width, label=schemes[schemes_index])
        autolabel(rects, ax)

    #     plt.title(schemes[schemes_index], fontsize=font_size)
    #     plt.xlabel("Reasons for Creating New Attacker", fontsize=font_size)
    #     plt.ylabel("number of attacker Created", fontsize=font_size)

    x_values = ["Be outside", "Honeypot"]
    ax.legend(prop={"size": legend_size})
    ax.set_xticks(temp_x)
    ax.set_xticklabels(x_values)
    ax.set_xlabel('Reasons for Creating New Attacker', fontsize=font_size)
    ax.set_ylabel('number of attacker Created', fontsize=font_size)
    plt.show()


# System Failure reason
def display_system_fail(schemes):
    fig, ax = plt.subplots(figsize=(figure_width, figure_high))
    width = 0.2
    shift_value = [- width / 2 - width, - width / 2, + width / 2, width / 2 + width]
    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/R_self_3/system_fail.pkl", "rb")
        SysFail_reason = pickle.load(the_file)
        the_file.close()

        print(SysFail_reason)
        y_values = SysFail_reason
        temp_x = np.arange(len(y_values))
        rects = ax.bar(temp_x + shift_value[schemes_index], y_values, width, label=schemes[schemes_index])
        autolabel(rects, ax)

    x_values = ["All node Evicted", "SF condition 1", "SF condition 2"]
    ax.legend(prop={"size": legend_size})
    ax.set_xticks(temp_x)
    ax.set_xticklabels(x_values)
    ax.set_xlabel("Reasons for System Failure", fontsize=font_size)
    ax.set_ylabel("number of simulation", fontsize=font_size)
    plt.show()


# NIDS Eviction
def display_NIDS_eviction(schemes):
    fig, ax = plt.subplots(figsize=(figure_width, figure_high))
    width = 0.2
    shift_value = [- width / 2 - width, - width / 2, + width / 2, width / 2 + width]
    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/R_self_4/NIDS_eviction.pkl", "rb")
        NIDS_eviction = pickle.load(the_file)
        the_file.close()

        y_scale = np.zeros(4)
        for key in NIDS_eviction.keys():
            y_scale += NIDS_eviction[key]

        # get average value
        y_scale = y_scale / len(NIDS_eviction.keys())

        # normalize to range(0, 1)
        y_scale = y_scale / (sum(y_scale))

        temp_x = np.arange(len(y_scale))
        rects = ax.bar(temp_x + shift_value[schemes_index], y_scale, width, label=schemes[schemes_index])
        autolabel(rects, ax)

    x_scale = ["Bad node\nevicted by\nIDS", "Good node\nevicted by\nIDS", "Bad node\nevicted by\nDS_4",
               "Good node\nevicted by\nDS_4"]
    ax.legend(prop={"size": legend_size})
    ax.set_xticks(temp_x)
    ax.set_xticklabels(x_scale, fontsize=font_size)
    ax.set_ylabel('Frequency', fontsize=font_size)
    plt.show()

# HNE hitting ratio
def display_hitting_prob(schemes):
    max_x_axis = 70
    plt.figure(figsize=(figure_width, figure_high))
    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/R_self_4/hitting_probability.pkl", "rb")
        hitting_prob = pickle.load(the_file)
        the_file.close()

        max_length = 0
        for key in hitting_prob.keys():
            if max_length < len(hitting_prob[key]):
                max_length = len(hitting_prob[key])

        if max_length > max_x_axis:
            max_length = max_x_axis

        hit_prob = []

        for index in range(max_length):
            hit_counter = 0
            total_counter = 0
            for key in hitting_prob.keys():
                if index < len(hitting_prob[key]):
                    total_counter += 1
                    if hitting_prob[key][index]:
                        hit_counter += 1
            hit_prob.append(hit_counter/total_counter)

        x_values = range(len(hit_prob))
        y_values = hit_prob
        plt.plot(x_values, y_values, linestyle=all_linestyle[schemes_index], label=schemes[schemes_index],
                     linewidth=figure_linewidth, marker=marker_list[schemes_index], markevery=50, markersize=marker_size)
    # plt.legend(prop={"size": legend_size},
    #            ncol=2,
    #            loc='upper left')
    # bbox_to_anchor = (-0.25, 1.01, 1.25, 0),
    plt.xlabel("number of games", fontsize=font_size)
    plt.ylabel("HNE hitting ratio", fontsize=font_size)
    plt.xticks(fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    plt.tight_layout()
    os.makedirs("Figure/hitting_ratio", exist_ok=True)
    os.makedirs("Figure (PNG)/hitting_ratio", exist_ok=True)
    plt.savefig("Figure/hitting_ratio/hit-ratio-VV.eps", dpi=figure_dpi)
    plt.savefig("Figure (PNG)/hitting_ratio/hit-ratio-VV.png", dpi=figure_dpi)
    plt.show()


def display_legend_scheme(schemes):
    fig = plt.figure()
    figlegend = plt.figure(figsize=(8.1, 0.6))
    ax = fig.add_subplot(111)


    lines = []
    for schemes_index in range(len(schemes)):
        line, = ax.plot([1, 2, 3], linestyle=all_linestyle[schemes_index], markersize=marker_size/1.5, marker=marker_list[schemes_index], label=schemes[schemes_index])
        lines.append(line)
    figlegend.legend(handles=lines,prop={"size": legend_size}, ncol=len(schemes))

    os.makedirs("Figure/legend", exist_ok=True)
    os.makedirs("Figure (PNG)/legend", exist_ok=True)
    plt.savefig("Figure/legend/the-legend.eps", dpi=figure_dpi)
    plt.savefig("Figure (PNG)/legend/the-legend.png", dpi=figure_dpi)
    fig.show()
    figlegend.show()


def display_legend_strat():
    fig = plt.figure()
    figlegend = plt.figure(figsize=(14.7, 0.6))
    ax = fig.add_subplot(111)


    lines = []
    for index in range(8):
        line, = ax.plot(np.arange(8), linestyle=all_linestyle[index], label=f"Stra {index + 1}")
        lines.append(line)
    figlegend.legend(handles=lines,prop={"size": legend_size}, ncol=8)
    # for index in range(8):
    #     plt.plot(range(max_length),
    #              average_att_strat[:, index],
    #              linestyle=all_linestyle[index],
    #              label=f"Stra {index + 1}")
    #
    # plt.legend(prop={"size": legend_size},
    #            ncol=4,
    #            bbox_to_anchor=(0, 1, 1, 0),
    #            loc='lower left',
    #            mode="expand")

    os.makedirs("Figure/legend", exist_ok=True)
    os.makedirs("Figure (PNG)/legend", exist_ok=True)
    plt.savefig("Figure/legend/the-legend_strat.eps", dpi=figure_dpi)
    plt.savefig("Figure (PNG)/legend/the-legend_start.png", dpi=figure_dpi)
    fig.show()
    figlegend.show()

# MTTSF
def sens_analy_MTTSF(schemes, sensitive_analysis, sen_ana_X_label):
    for value_name, x_label in zip(sensitive_analysis, sen_ana_X_label):
        plt.figure(figsize=(figure_width, figure_high))
        for schemes_index in range(len(schemes)):
            the_file = open("data/" + schemes[schemes_index] + "/"+value_name+"/MTTSF.pkl", "rb")
            MTTSF = pickle.load(the_file)
            the_file.close()
            the_file = open("data/" + schemes[schemes_index] + "/"+value_name+"/Range.pkl", "rb")
            vary_range = pickle.load(the_file)
            the_file.close()

            plt.plot(vary_range,
                     MTTSF,
                     linestyle=all_linestyle[schemes_index],
                     label=schemes[schemes_index],
                     linewidth=figure_linewidth,
                     marker=marker_list[schemes_index], markersize=marker_size)

        # plt.legend(prop={"size":legend_size}, ncol=2, bbox_to_anchor=(0, 1, 1, 0),
        #               loc='lower left', fontsize='large',mode="expand")
        plt.xlabel("Varying "+x_label, fontsize=font_size)
        plt.ylabel("MTTSF", fontsize=font_size)
        plt.xticks(fontsize=axis_size)
        plt.yticks(fontsize=axis_size)
        plt.tight_layout()
        os.makedirs("Figure/"+value_name, exist_ok=True)
        os.makedirs("Figure (PNG)/"+value_name, exist_ok=True)
        plt.savefig("Figure/"+value_name+"/MTTSF_"+value_name+".eps", dpi=figure_dpi)
        plt.savefig("Figure (PNG)/"+value_name+"/MTTSF_"+value_name+".png", dpi=figure_dpi)
        plt.show()


# attacker Uncertainty varying variables
def sens_analy_att_uncertain(schemes, sensitive_analysis, sen_ana_X_label):
    for value_name, x_label in zip(sensitive_analysis, sen_ana_X_label):
        plt.figure(figsize=(figure_width, figure_high))
        for schemes_index in range(len(schemes)):
            the_file = open("data/" + schemes[schemes_index] + "/"+value_name+"/attacker_uncertainty.pkl", "rb")
            att_uncertainty_all_result = pickle.load(the_file)
            the_file.close()
            the_file = open("data/" + schemes[schemes_index] + "/" + value_name + "/Range.pkl", "rb")
            vary_range = pickle.load(the_file)
            the_file.close()


            plt.plot(vary_range,
                     att_uncertainty_all_result,
                     linestyle=all_linestyle[schemes_index],
                     label=schemes[schemes_index],
                     linewidth=figure_linewidth,
                     marker=marker_list[schemes_index], markersize=marker_size)

        # plt.legend(prop={"size":15}, ncol=4, bbox_to_anchor=(0, 1, 1, 0),
        #               loc='lower left', fontsize='large',mode="expand")
        plt.xlabel("Varying "+x_label, fontsize=font_size)
        plt.ylabel("Uncertaity", fontsize=font_size)
        plt.xticks(fontsize=axis_size)
        plt.yticks(fontsize=axis_size)
        plt.tight_layout()
        os.makedirs("Figure/"+value_name, exist_ok=True)
        os.makedirs("Figure (PNG)/"+value_name, exist_ok=True)
        plt.savefig("Figure/"+value_name+"/att-uncertain_"+value_name+".eps", dpi=figure_dpi)
        plt.savefig("Figure (PNG)/"+value_name+"/att-uncertain_"+value_name+".png", dpi=figure_dpi)
        plt.show()

# defender Uncertainty varying variables
def sens_analy_def_uncertain(schemes, sensitive_analysis, sen_ana_X_label):
    for value_name, x_label in zip(sensitive_analysis, sen_ana_X_label):
        plt.figure(figsize=(figure_width, figure_high))
        for schemes_index in range(len(schemes)):
            the_file = open("data/" + schemes[schemes_index] + "/"+value_name+"/defender_uncertainty.pkl", "rb")
            def_uncertainty_all_result = pickle.load(the_file)
            the_file.close()
            the_file = open("data/" + schemes[schemes_index] + "/" + value_name + "/Range.pkl", "rb")
            vary_range = pickle.load(the_file)
            the_file.close()


            plt.plot(vary_range,
                     def_uncertainty_all_result,
                     linestyle=all_linestyle[schemes_index],
                     label=schemes[schemes_index],
                     linewidth=figure_linewidth,
                     marker=marker_list[schemes_index], markersize=marker_size)

        # plt.legend(prop={"size":15}, ncol=4, bbox_to_anchor=(0, 1, 1, 0),
        #               loc='lower left', fontsize='large',mode="expand")
        plt.xlabel("Varying "+x_label, fontsize=font_size)
        plt.ylabel("Uncertaity", fontsize=font_size)
        plt.xticks(fontsize=axis_size)
        plt.yticks(fontsize=axis_size)
        plt.tight_layout()
        os.makedirs("Figure/"+value_name, exist_ok=True)
        os.makedirs("Figure (PNG)/"+value_name, exist_ok=True)
        plt.savefig("Figure/"+value_name+"/def-uncertain_"+value_name+".eps", dpi=figure_dpi)
        plt.savefig("Figure (PNG)/"+value_name+"/def-uncertain_"+value_name+".png", dpi=figure_dpi)
        plt.show()

# defender strategy success or fail counter
def defstrat_success_counter_DDIPI():
    the_file = open("data/DD-IPI/R6/def_succ_counter.pkl", "rb")
    def_succ_counter_all_result = pickle.load(the_file)
    the_file.close()

    the_file = open("data/DD-IPI/R6/def_fail_counter.pkl", "rb")
    def_fail_counter_all_result = pickle.load(the_file)
    the_file.close()

    plt.figure(figsize=(figure_width, figure_high + 1))
    succ_counter_sum = np.zeros((6,8))
    fail_counter_sum = np.zeros((6,8))

    for game_id in range(len(def_succ_counter_all_result)):
        def_succ_counter = def_succ_counter_all_result[game_id]
        def_fail_counter = def_fail_counter_all_result[game_id]
        succ_counter_sum += def_succ_counter
        fail_counter_sum += def_fail_counter

    # only consider inside attacker
    succ_counter = np.zeros(8)
    fail_counter = np.zeros(8)

    start_CKC = 2
    for CKC_id in np.arange(start_CKC,len(succ_counter_sum)):
        succ_counter += succ_counter_sum[CKC_id]
        fail_counter += fail_counter_sum[CKC_id]

    plt.bar(["DS"+str(strat_ID) for strat_ID in np.arange(8)+1], fail_counter)

    plt.xlabel("Defense Strategy", fontsize=font_size)
    plt.ylabel("Number of attack failed ", fontsize=font_size)
    plt.xticks(fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    plt.tight_layout()
    os.makedirs("Figure/def_for_review", exist_ok=True)
    os.makedirs("Figure (PNG)/def_for_review", exist_ok=True)
    plt.savefig("Figure/def_for_review/def_cause_att_fail.eps", dpi=figure_dpi)
    plt.savefig("Figure (PNG)/def_for_review/def_cause_att_fail.png", dpi=figure_dpi)
    plt.show()


def def_cost_per_strategy_DDIPI():
    the_file = open("data/DD-PI/R6/def_cost_per_strat.pkl", "rb")
    def_per_strat_cost_allresult = pickle.load(the_file)
    the_file.close()

    plt.figure(figsize=(figure_width, figure_high+0.75))

    max_length = 0
    for key in def_per_strat_cost_allresult.keys():
        if max_length < len(def_per_strat_cost_allresult[key]):
            max_length = len(def_per_strat_cost_allresult[key])

    average_def_cost = np.zeros((1,8))
    for index in range(max_length):
        sum_on_index = np.zeros(8)
        number_on_index = 0
        for key in def_per_strat_cost_allresult.keys():
            if len(def_per_strat_cost_allresult[key]) > 0:
                sum_on_index += def_per_strat_cost_allresult[key][0]
                def_per_strat_cost_allresult[key] = np.delete(def_per_strat_cost_allresult[key],0,0)
                number_on_index += 1

        average_def_cost = np.append(average_def_cost, np.reshape(sum_on_index / number_on_index, (1, -1)), axis=0)

    average_def_cost = average_def_cost[1:100]
    average_def_cost[0,7] = 0
    for strat_id in range(4,average_def_cost.shape[1]):
        plt.plot(np.arange(average_def_cost.shape[0]), average_def_cost[:,strat_id],
                     label="DS"+str(strat_id+1),
                     linewidth=figure_linewidth)
    plt.legend(prop={"size": 15}, ncol=4, bbox_to_anchor=(0, 1, 1, 0), loc='lower left', fontsize='large',mode="expand")

    plt.xlabel("number of games", fontsize=font_size)
    plt.ylabel("Defense Cost ", fontsize=font_size)
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    plt.xticks(fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    axes = plt.gca()
    axes.set_ylim([0, 0.5])
    plt.tight_layout()
    os.makedirs("Figure/def_for_review", exist_ok=True)
    os.makedirs("Figure (PNG)/def_for_review", exist_ok=True)
    plt.savefig("Figure/def_for_review/def_cost_per_strat_DDPI.eps", dpi=figure_dpi)
    plt.savefig("Figure (PNG)/def_for_review/def_cost_per_strat_DDPI.png", dpi=figure_dpi)
    plt.show()




# att_cost_all_result = np.zeros(8)
#
# value_range = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
# scheme_range = ["DD-IPI", "DD-PI", "No-DD-IPI", "No-DD-PI"]
# for scheme in scheme_range:
#     for index in range(len(value_range)):
#         the_file = open("data/best_NIDS/"+scheme+"/"+str(value_range[index])+"/att_cost.pkl",
#                             "rb")
#         att_cost_all_result[index] = pickle.load(the_file)
#         the_file.close()
#
#     plt.plot(value_range, att_cost_all_result, label=scheme)
#
# plt.legend(prop={"size":legend_size})
# plt.xlabel("NIDS threshold")
# plt.ylabel("Attacker Cost")
#
#
#
#
# def_cost_all_result = np.zeros(8)
#
# value_range = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
# scheme_range = ["DD-IPI", "DD-PI", "No-DD-IPI", "No-DD-PI"]
# for scheme in scheme_range:
#     for index in range(len(value_range)):
#         the_file = open("data/best_NIDS/"+scheme+"/"+str(value_range[index])+"/def_cost.pkl",
#                             "rb")
#         def_cost_all_result[index] = pickle.load(the_file)
#         the_file.close()
#
#     plt.plot(value_range, def_cost_all_result, label=scheme)
#
# plt.legend(prop={"size":legend_size})
# plt.xlabel("NIDS threshold")
# plt.ylabel("Defender Cost")
#
#
#
#
# MTTSF_all_result = np.zeros(8)
#
# value_range = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
# scheme_range = ["DD-IPI", "DD-PI", "No-DD-IPI", "No-DD-PI"]
# for scheme in scheme_range:
#     for index in range(len(value_range)):
#         the_file = open("data/best_NIDS/"+scheme+"/"+str(value_range[index])+"/MTTSF.pkl",
#                             "rb")
#         MTTSF_all_result[index] = pickle.load(the_file)
#         the_file.close()
#
#     plt.plot(value_range, MTTSF_all_result, label=scheme)
#
# plt.legend(prop={"size":legend_size})
# plt.xlabel("NIDS threshold")
# plt.ylabel("MTTSF")


if __name__ == '__main__':
    all_linestyle = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    font_size = 25
    figure_high = 4
    figure_width = 7.5
    figure_linewidth = 3
    figure_dpi = 200
    legend_size = 17
    axis_size = 20
    marker_size = 10
    marker_list = ["p", "d", "v", "x", "s", "*", "1", "."]

    os.makedirs("Figure/VUB", exist_ok=True)
    os.makedirs("Figure (PNG)/VUB", exist_ok=True)
    sensitive_analysis = ["Th_risk", "_lambda", "mu", "SF_thres_1", "SF_thres_2", "att_detect_UpBod"]
    sen_ana_X_label = ["$Th_{risk}$", "$\lambda$", "$\mu$", r"$\rho_1$", r"$\rho_2$", "ad"]

    schemes = ["DD-IPI", "DD-PI", "No-DD-IPI", "No-DD-PI"]
    # schemes = ["DD-IPI"]

    # display_FPR(schemes)
    # display_TPR(schemes)
    # display_hitting_prob(schemes)
    # display_legend_scheme(schemes)
    # display_legend_strat()
    # display_MTTSF(schemes)
    # varying_vul_att_cost(schemes)
    # varying_vul_def_cost(schemes)
    # varying_vul_att_HEU(schemes)
    # varying_vul_def_HEU(schemes)
    # varying_vul_att_uncertain(schemes)
    # varying_vul_def_uncertain(schemes)
    # varying_vul_FPR(schemes)
    # varying_vul_TPR(schemes)
    # defstrat_success_counter_DDIPI()
    # def_cost_per_strategy_DDIPI()
    # DD_IPI_def_strat_count()
    # DD_PI_att_strat_count()
    No_DD_PI_att_strat_count()


    # sens_analy_MTTSF(schemes, sensitive_analysis, sen_ana_X_label)
    # sens_analy_att_uncertain(schemes, sensitive_analysis, sen_ana_X_label)
    # sens_analy_def_uncertain(schemes, sensitive_analysis, sen_ana_X_label)

