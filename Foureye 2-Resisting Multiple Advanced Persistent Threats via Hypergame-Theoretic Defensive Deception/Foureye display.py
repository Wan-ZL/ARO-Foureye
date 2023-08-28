import math
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import MaxNLocator
import textwrap
import pylab


def display_TTSF():
    # schemes = ["DD-IPI", "DD-Random-IPI","DD-ML-IPI", "DD-PI", "DD-Random-PI","DD-ML-PI"]

    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/R0/Time_to_SF.pkl", "rb")
        # the_file = open("data/" + current_scheme + "/R0/Time_to_SF.pkl", "rb")
        TTSF = pickle.load(the_file)
        # TTSF_list = sorted(list(TTSF.values()))
        TTSF_list = list(TTSF.values())

        plt.figure(figsize=(figure_width, figure_high))
        plt.bar(range(len(TTSF_list)), TTSF_list, hatch=patterns[schemes_index])
        plt.axhline(y=np.mean(TTSF_list), color='r', linestyle=':')

        plt.xlabel("Simulation ID", fontsize=font_size)
        plt.ylabel("TTSF", fontsize=font_size)
        plt.xticks(fontsize=axis_size)
        plt.yticks(fontsize=axis_size)
        plt.tight_layout()
        os.makedirs("Figure/" + schemes[schemes_index], exist_ok=True)
        plt.savefig("Figure/" + schemes[schemes_index] + "/TTSF.svg", dpi=figure_dpi)
        plt.savefig("Figure/" + schemes[schemes_index] + "/TTSF.png", dpi=figure_dpi)
        plt.show()

def display_TTSF_in_one():
    # schemes = ["DD-IPI", "DD-Random-IPI","DD-ML-IPI", "DD-PI", "DD-Random-PI","DD-ML-PI"]

    # plt.figure(figsize=(figure_width, figure_high))
    fig, ax = plt.subplots(figsize=(figure_width, figure_high))
    width = 0.8
    bar_group_number = len(schemes)
    notation = -1
    coloar_order = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/R0/Time_to_SF.pkl", "rb")
        # the_file = open("data/" + current_scheme + "/R0/Time_to_SF.pkl", "rb")
        TTSF = pickle.load(the_file)
        # TTSF_list = sorted(list(TTSF.values()))
        TTSF_list = list(TTSF.values())

        x = np.arange(len(TTSF_list))
        # plt.bar(range(len(TTSF_list)), TTSF_list, hatch=patterns[schemes_index])
        # print(x + notation * width/bar_group_number)
        # print(width/bar_group_number)
        ax.bar(x + notation * width/bar_group_number, TTSF_list, width/bar_group_number, label=schemes[schemes_index], hatch=patterns[schemes_index])
        ax.axhline(y=np.mean(TTSF_list), color=coloar_order[schemes_index], linestyle='-.')
        print(schemes[schemes_index]+f" {np.mean(TTSF_list)}")
        notation += 1

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend()
    plt.xlabel("Simulation ID", fontsize=font_size)
    plt.ylabel("TTSF", fontsize=font_size)
    plt.xticks(fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    plt.tight_layout()
    os.makedirs("Figure/All-In-One", exist_ok=True)
    plt.savefig("Figure/All-In-One/TTSF_all_in_one.svg", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/TTSF_all_in_one.png", dpi=figure_dpi)
    plt.show()


def display_TTSF_in_one_bar():
    # schemes = ["DD-IPI", "DD-Random-IPI","DD-ML-IPI", "DD-PI", "DD-Random-PI","DD-ML-PI"]
    # schemes = ["DD-IPI", "DD-Random-IPI", "DD-PI"]

    plt.figure(figsize=(figure_width, figure_high))
    # fig, ax = plt.subplots(figsize=(figure_width, figure_high))
    width = 0.8
    bar_group_number = len(schemes)
    notation = -1
    coloar_order = plt.rcParams['axes.prop_cycle'].by_key()['color']
    y_result_list = []
    box_plot_set = []
    error = []
    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/R0/Time_to_SF.pkl", "rb")
        # the_file = open("data/" + current_scheme + "/R0/Time_to_SF.pkl", "rb")
        TTSF = pickle.load(the_file)
        # TTSF_list = sorted(list(TTSF.values()))
        TTSF_list = list(TTSF.values())
        box_plot_set.append(TTSF_list)
        y_result_list.append(np.mean(TTSF_list))
        error.append(np.std(TTSF_list))
        # plt.bar(range(len(TTSF_list)), TTSF_list, hatch=patterns[schemes_index])
        # print(x + notation * width/bar_group_number)
        # print(width/bar_group_number)
        # ax.bar(x + notation * width/bar_group_number, TTSF_list, width/bar_group_number, label=schemes[schemes_index], hatch=patterns[schemes_index])
        # ax.axhline(y=np.mean(TTSF_list), color=coloar_order[schemes_index], linestyle='-.')
        print(schemes[schemes_index]+f" {np.mean(TTSF_list)}")
        notation += 1

    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # plt.boxplot(box_plot_set, positions=range(len(box_plot_set)), whis=1)
    for index in np.arange(len(y_result_list)):
        print(index)
        plt.bar(index, y_result_list[index], yerr=error[index], capsize=10, align='center', hatch=patterns[index])
    # for x in range(len(y_result_list)):
        # plt.text(x+0.03, y_result_list[x]+0.5 , round(y_result_list[x],2))

    plt.xticks(np.arange(len(schemes)), [textwrap.fill(label, 7) for label in legend_name], fontsize=axis_size*0.6)
    plt.yticks(fontsize=axis_size)
    plt.xlabel("Schemes", fontsize=font_size)
    plt.ylabel("MTTSF", fontsize=font_size)
    plt.tight_layout()
    os.makedirs("Figure/All-In-One", exist_ok=True)
    plt.savefig("Figure/All-In-One/TTSF_all_in_one_bar.svg", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/TTSF_all_in_one_bar.png", dpi=figure_dpi)
    plt.show()


def display_TTSF_vary_AttArivalProb():
    folder_list = ["data (1)", "data (2)", "data (3)", "data (4)", "data (5)"]
    vary_parameter = [0.1, 0.2, 0.3, 0.4, 0.5]
    plt.figure(figsize=(figure_width, figure_high))
    display_dataset = np.zeros((len(folder_list), len(schemes)))
    error = np.zeros((len(folder_list), len(schemes)))
    for folder_index in range(len(folder_list)):
        for schemes_index in range(len(schemes)):
            the_file = open("data/vary_parameter/MTTSF_AttArivalProb/"+ folder_list[folder_index] +"/"+ schemes[schemes_index] + "/R0/Time_to_SF.pkl", "rb")
            TTSF = pickle.load(the_file)
            # TTSF_list = sorted(list(TTSF.values()))
            TTSF_list = list(TTSF.values())
            display_dataset[folder_index, schemes_index] = np.mean(TTSF_list)
            error[folder_index, schemes_index] = np.std(TTSF_list)

    for scheme_index in range(len(schemes)):
        plt.plot(vary_parameter, display_dataset[:,scheme_index], label=schemes[scheme_index])

    plt.legend()
    plt.xlabel("Attacker Arival Probability", fontsize=font_size)
    plt.ylabel("MTTSF", fontsize=font_size)
    plt.tight_layout()
    os.makedirs("data/vary_parameter/MTTSF_AttArivalProb", exist_ok=True)
    plt.savefig("data/vary_parameter/MTTSF_AttArivalProb/MTTSF_vary_AttArivalProb.svg", dpi=figure_dpi)
    plt.savefig("data/vary_parameter/MTTSF_AttArivalProb/MTTSF_vary_AttArivalProb.png", dpi=figure_dpi)
    plt.show()


def display_HEU_In_One():
    # schemes = ["DD-IPI", "DD-Random-IPI","DD-ML-IPI", "DD-PI", "DD-Random-PI","DD-ML-PI"]
    # AHEU
    plt.figure(figsize=(figure_width, figure_high))
    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/R1/att_HEU.pkl", "rb")
        att_HEU_all_result = pickle.load(the_file)
        the_file.close()

        max_length = 0
        for dict_index in range(len(att_HEU_all_result)):
            if len(att_HEU_all_result[dict_index]) > max_length:
                max_length = len(att_HEU_all_result[dict_index])

        summed_AHEU_list = np.zeros(max_length)
        denominator_list = np.zeros(max_length)
        for key in att_HEU_all_result.keys():
            summed_AHEU_list[:len(att_HEU_all_result[key])] += [np.mean(k) if k.size > 0 else 0 for k in
                                                                att_HEU_all_result[key]]
            denominator_list[:len(att_HEU_all_result[key])] += np.ones(len(att_HEU_all_result[key]))

        averaged_AHEU_list = summed_AHEU_list / denominator_list
        plt.plot(range(1, len(averaged_AHEU_list)), averaged_AHEU_list[1:], linestyle=all_linestyle[schemes_index], label=legend_name[schemes_index])

    plt.legend(ncol=2)
    plt.xlabel("# of games", fontsize=font_size)
    plt.ylabel("Averaged C-AHEU", fontsize=font_size)
    plt.xticks(fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    plt.xlim([0, max_x_length])  # fix x axis range
    plt.tight_layout()
    os.makedirs("Figure/All-In-One", exist_ok=True)
    plt.savefig("Figure/All-In-One/AHEU_AllInOne.svg", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/AHEU_AllInOne.png", dpi=figure_dpi)
    plt.show()

    # DHEU
    plt.figure(figsize=(figure_width, figure_high))
    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/R1/def_HEU.pkl", "rb")
        def_HEU_all_result = pickle.load(the_file)
        the_file.close()

        max_length = 0
        for dict_index in range(len(def_HEU_all_result)):
            if len(def_HEU_all_result[dict_index]) > max_length:
                max_length = len(def_HEU_all_result[dict_index])

        summed_DHEU_list = np.zeros(max_length)
        denominator_list = np.zeros(max_length)
        for key in def_HEU_all_result.keys():
            summed_DHEU_list[:len(def_HEU_all_result[key])] += [np.mean(k) for k in def_HEU_all_result[key]]
            denominator_list[:len(def_HEU_all_result[key])] += np.ones(len(def_HEU_all_result[key]))

        averaged_DHEU_list = summed_DHEU_list / denominator_list

        plt.plot(range(1, len(averaged_DHEU_list)), averaged_DHEU_list[1:], linestyle=all_linestyle[schemes_index], label=legend_name[schemes_index])

    plt.legend()
    plt.xlabel("# of games", fontsize=font_size)
    plt.ylabel("Averaged C-DHEU", fontsize=font_size)
    plt.xticks(fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    plt.xlim([0, max_x_length])  # fix x axis range
    plt.tight_layout()
    os.makedirs("Figure/All-In-One", exist_ok=True)
    plt.savefig("Figure/All-In-One/DHEU_AllInOne.svg", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/DHEU_AllInOne.png", dpi=figure_dpi)
    plt.show()

def display_average_HEU_In_One():
    # schemes = ["DD-IPI", "DD-Random-IPI","DD-ML-IPI", "DD-PI", "DD-Random-PI","DD-ML-PI"]
    # AHEU
    plt.figure(figsize=(figure_width, figure_high))
    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/R1/att_HEU.pkl", "rb")
        att_HEU_all_result = pickle.load(the_file)
        the_file.close()

        max_length = 0
        for dict_index in range(len(att_HEU_all_result)):
            if len(att_HEU_all_result[dict_index]) > max_length:
                max_length = len(att_HEU_all_result[dict_index])

        summed_AHEU_list = np.zeros(max_length)
        denominator_list = np.zeros(max_length)
        for key in att_HEU_all_result.keys():
            summed_AHEU_list[:len(att_HEU_all_result[key])] += [np.mean(k) if k.size > 0 else 0 for k in
                                                                att_HEU_all_result[key]]
            denominator_list[:len(att_HEU_all_result[key])] += np.ones(len(att_HEU_all_result[key]))

        averaged_AHEU_list = summed_AHEU_list / denominator_list
        plt.bar(schemes_index, np.mean(averaged_AHEU_list), hatch=patterns[schemes_index], label=schemes[schemes_index])
        plt.text(schemes_index - 0.2, np.mean(averaged_AHEU_list) + 0.05, round(np.mean(averaged_AHEU_list), 2))
        # plt.plot(range(1, len(averaged_AHEU_list)), averaged_AHEU_list[1:], linestyle=all_linestyle[schemes_index], label=schemes[schemes_index])

    # plt.legend()
    plt.xlabel("Schemes", fontsize=font_size)
    plt.ylabel("Average C-AHEU", fontsize=font_size)
    plt.xticks(range(len(schemes) + 1), [textwrap.fill(label, 7) for label in schemes], fontsize=0.6*axis_size)
    plt.yticks(fontsize=axis_size)
    plt.tight_layout()
    os.makedirs("Figure/All-In-One", exist_ok=True)
    plt.savefig("Figure/All-In-One/average_AHEU_AllInOne.svg", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/average_AHEU_AllInOne.png", dpi=figure_dpi)
    plt.show()

    # DHEU
    plt.figure(figsize=(figure_width, figure_high))
    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/R1/def_HEU.pkl", "rb")
        def_HEU_all_result = pickle.load(the_file)
        the_file.close()

        max_length = 0
        for dict_index in range(len(def_HEU_all_result)):
            if len(def_HEU_all_result[dict_index]) > max_length:
                max_length = len(def_HEU_all_result[dict_index])

        summed_DHEU_list = np.zeros(max_length)
        denominator_list = np.zeros(max_length)
        for key in def_HEU_all_result.keys():
            summed_DHEU_list[:len(def_HEU_all_result[key])] += [np.mean(k) for k in def_HEU_all_result[key]]
            denominator_list[:len(def_HEU_all_result[key])] += np.ones(len(def_HEU_all_result[key]))

        averaged_DHEU_list = summed_DHEU_list / denominator_list

        plt.bar(schemes_index, np.mean(averaged_DHEU_list), hatch=patterns[schemes_index], label=schemes[schemes_index])
        plt.text(schemes_index - 0.2, np.mean(averaged_DHEU_list) + 0.05, round(np.mean(averaged_DHEU_list), 2))

    # plt.legend()
    plt.xlabel("Schemes", fontsize=font_size)
    plt.ylabel("Average C-DHEU", fontsize=font_size)
    plt.xticks(range(len(schemes) + 1), [textwrap.fill(label, 7) for label in schemes],fontsize=0.6*axis_size)
    plt.yticks(fontsize=axis_size)
    plt.tight_layout()
    os.makedirs("Figure/All-In-One", exist_ok=True)
    plt.savefig("Figure/All-In-One/average_DHEU_AllInOne.svg", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/average_DHEU_AllInOne.png", dpi=figure_dpi)
    plt.show()


def display_strategy_count():
    # schemes = ["DD-IPI", "DD-Random-IPI","DD-ML-IPI", "DD-PI", "DD-Random-PI","DD-ML-PI"]


    # AHEU
    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/R2/att_strategy_counter.pkl", "rb")
        att_strategy = pickle.load(the_file)
        the_file.close()

        max_length = 0
        for key in att_strategy.keys():
            if len(att_strategy[key]) > max_length:
                max_length = len(att_strategy[key])

        attack_strategy_counter = np.zeros((strategy_number, max_length))  # 9 is the number of strategies
        for key in att_strategy.keys():
            for strategy_id in range(strategy_number):
                attack_strategy_counter[strategy_id, :len(att_strategy[key])] += [k.count(strategy_id) for k in
                                                                                  att_strategy[key]]

        strategy_sum_in_each_game = [sum(k) for k in attack_strategy_counter.T]

        # AHEU By Number
        plt.figure(figsize=(figure_width, figure_high))
        for strategy_id in range(strategy_number):
            plt.plot(range(max_length), attack_strategy_counter[strategy_id], label=f"Stra {strategy_id + 1}")
        plt.legend(prop={"size": legend_size}, ncol=4, bbox_to_anchor=(0, 1, 1, 0), loc='lower left', mode="expand")
        plt.xlabel("number of games ("+schemes[schemes_index]+")", fontsize=font_size)
        plt.ylabel("Number of Att strategy used", fontsize=font_size / 1.5)
        plt.xticks(fontsize=axis_size)
        plt.yticks(fontsize=axis_size)
        plt.tight_layout()
        os.makedirs("Figure/" + schemes[schemes_index], exist_ok=True)
        plt.savefig("Figure/" + schemes[schemes_index] + "/att-Strat-in-Number.svg", dpi=figure_dpi)
        plt.savefig("Figure/" + schemes[schemes_index] + "/att-Strat-in-Number.png", dpi=figure_dpi)
        plt.show()

        # AHEU By Percentage
        plt.figure(figsize=(figure_width, figure_high))
        percentage_style_attack_strategy_counter = attack_strategy_counter / strategy_sum_in_each_game
        for strategy_id in range(strategy_number):
            plt.plot(range(max_length), percentage_style_attack_strategy_counter[strategy_id],
                     label=f"Stra {strategy_id + 1}")
        plt.legend(prop={"size": legend_size}, ncol=4, bbox_to_anchor=(0, 1, 1, 0), loc='lower left', mode="expand")
        plt.xlabel("number of games ("+schemes[schemes_index]+")", fontsize=font_size)
        plt.ylabel("Percentage of Att strategy used", fontsize=font_size / 1.5)
        plt.xticks(fontsize=axis_size)
        plt.yticks(fontsize=axis_size)
        plt.tight_layout()
        plt.savefig("Figure/" + schemes[schemes_index] + "/att-Strat-in-Percentage.svg", dpi=figure_dpi)
        plt.savefig("Figure/" + schemes[schemes_index] + "/att-Strat-in-Percentage.png", dpi=figure_dpi)
        plt.show()

    # DHEU
    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/R2/def_strategy_counter.pkl", "rb")
        def_strategy = pickle.load(the_file)
        the_file.close()

        max_length = 0
        for key in def_strategy.keys():
            if len(def_strategy[key]) > max_length:
                max_length = len(def_strategy[key])

        defend_strategy_counter = np.zeros((strategy_number, max_length))  # 9 is the number of strategies
        for key in def_strategy.keys():
            for strategy_id in range(strategy_number):
                defend_strategy_counter[strategy_id, :len(def_strategy[key])] += [k.count(strategy_id) for k in
                                                                                  def_strategy[key]]

        strategy_sum_in_each_game = [sum(k) for k in defend_strategy_counter.T]

        # DHEU By Number
        plt.figure(figsize=(figure_width, figure_high))
        for strategy_id in range(strategy_number):
            plt.plot(range(max_length), defend_strategy_counter[strategy_id], label=f"Stra {strategy_id + 1}")
        plt.legend(prop={"size": legend_size}, ncol=4, bbox_to_anchor=(0, 1, 1, 0), loc='lower left', mode="expand")
        plt.xlabel("number of games ("+schemes[schemes_index]+")", fontsize=font_size)
        plt.ylabel("Number of Def strategy used", fontsize=font_size / 1.5)
        plt.xticks(fontsize=axis_size)
        plt.yticks(fontsize=axis_size)
        plt.tight_layout()
        os.makedirs("Figure/" + schemes[schemes_index], exist_ok=True)
        plt.savefig("Figure/" + schemes[schemes_index] + "/def-Strat-in-Number.svg", dpi=figure_dpi)
        plt.savefig("Figure/" + schemes[schemes_index] + "/def-Strat-in-Number.png", dpi=figure_dpi)
        plt.show()

        # DHEU By Percentage
        plt.figure(figsize=(figure_width, figure_high))
        percentage_style_defend_strategy_counter = defend_strategy_counter / strategy_sum_in_each_game
        for strategy_id in range(strategy_number):
            plt.plot(range(max_length), percentage_style_defend_strategy_counter[strategy_id],
                     label=f"Stra {strategy_id + 1}")
        plt.legend(prop={"size": legend_size}, ncol=4, bbox_to_anchor=(0, 1, 1, 0), loc='lower left', mode="expand")
        plt.xlabel("number of games ("+schemes[schemes_index]+")", fontsize=font_size)
        plt.ylabel("Percentage of Def strategy used", fontsize=font_size / 1.5)
        plt.xticks(fontsize=axis_size)
        plt.yticks(fontsize=axis_size)
        plt.tight_layout()
        plt.savefig("Figure/" + schemes[schemes_index] + "/def-Strat-in-Percentage.svg", dpi=figure_dpi)
        plt.savefig("Figure/" + schemes[schemes_index] + "/def-Strat-in-Percentage.png", dpi=figure_dpi)
        plt.show()

def display_strategy_prob_distribution():
    # schemes = ["DD-IPI", "DD-Random-IPI","DD-ML-IPI", "DD-PI", "DD-Random-PI","DD-ML-PI"]

    # attacker
    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/R2/att_strategy_counter.pkl", "rb")
        att_strategy = pickle.load(the_file)
        the_file.close()

        strategy_counter = np.zeros(strategy_number)
        for key in att_strategy.keys():
            for strat_list in att_strategy[key]:
                for strat_id in strat_list:
                    strategy_counter[strat_id] += 1

        strategy_probability = strategy_counter/np.sum(strategy_counter)

        plt.figure(figsize=(figure_width, figure_high))
        plt.bar(range(1, len(strategy_probability)+1), strategy_probability, hatch=patterns[schemes_index])
        plt.title(schemes[schemes_index], fontsize=font_size)
        plt.xlabel("Attack Strategy ID", fontsize=font_size)
        plt.ylabel("Probability", fontsize=font_size / 1.5)
        plt.xticks(range(1, len(strategy_probability)+1), fontsize=axis_size)
        plt.yticks(fontsize=axis_size)
        plt.tight_layout()
        os.makedirs("Figure/" + schemes[schemes_index], exist_ok=True)
        plt.savefig("Figure/" + schemes[schemes_index] + "/att-Strat-prob-distribution.svg", dpi=figure_dpi)
        plt.savefig("Figure/" + schemes[schemes_index] + "/att-Strat-prob-distribution.png", dpi=figure_dpi)
        plt.show()


    # DHEU
    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/R2/def_strategy_counter.pkl", "rb")
        def_strategy = pickle.load(the_file)
        the_file.close()

        strategy_counter = np.zeros(strategy_number)
        for key in def_strategy.keys():
            for strat_list in def_strategy[key]:
                for strat_id in strat_list:
                    strategy_counter[strat_id] += 1

        strategy_probability = strategy_counter / np.sum(strategy_counter)

        plt.figure(figsize=(figure_width, figure_high))
        plt.bar(range(1, len(strategy_probability) + 1), strategy_probability, hatch=patterns[schemes_index])
        plt.title(schemes[schemes_index], fontsize=font_size)
        plt.xlabel("Defense Strategy ID", fontsize=font_size)
        plt.ylabel("Probability", fontsize=font_size / 1.5)
        plt.xticks(range(1, len(strategy_probability) + 1), fontsize=axis_size)
        plt.yticks(fontsize=axis_size)
        plt.tight_layout()
        os.makedirs("Figure/" + schemes[schemes_index], exist_ok=True)
        plt.savefig("Figure/" + schemes[schemes_index] + "/def-Strat-prob-distribution.svg", dpi=figure_dpi)
        plt.savefig("Figure/" + schemes[schemes_index] + "/def-Strat-prob-distribution.png", dpi=figure_dpi)
        plt.show()

def display_strategy_prob_distribution_in_one():
    # schemes = ["DD-IPI", "DD-Random-IPI","DD-ML-IPI", "DD-PI", "DD-Random-PI","DD-ML-PI"]

    fig, ax = plt.subplots(figsize=(figure_width, figure_high))
    width = 0.8
    bar_group_number = len(schemes)
    notation = -1
    # attacker
    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/R2/att_strategy_counter.pkl", "rb")
        att_strategy = pickle.load(the_file)
        the_file.close()

        strategy_counter = np.zeros(strategy_number)
        for key in att_strategy.keys():
            for strat_list in att_strategy[key]:
                for strat_id in strat_list:
                    strategy_counter[strat_id] += 1

        strategy_probability = strategy_counter/np.sum(strategy_counter)

        x = np.arange(1,len(strategy_probability)+1)
        ax.bar(x + notation * width / bar_group_number, strategy_probability, width / bar_group_number, label=schemes[schemes_index], hatch=patterns[schemes_index])
        notation += 1
        # plt.bar(range(1, len(strategy_probability)+1), strategy_probability, hatch=patterns[schemes_index])
    plt.legend()
    # plt.title(schemes, fontsize=font_size)
    plt.xlabel("Attack Strategy ID", fontsize=font_size)
    plt.ylabel("Probability", fontsize=font_size / 1.5)
    plt.xticks(range(1, len(strategy_probability)+1), fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    plt.tight_layout()
    os.makedirs("Figure/All-In-One", exist_ok=True)
    plt.savefig("Figure/All-In-One/att-Strat-prob-distribution_AllInOne.svg", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/att-Strat-prob-distribution_AllInOne.png", dpi=figure_dpi)
    plt.show()


    # DHEU
    notation = -1
    fig, ax = plt.subplots(figsize=(figure_width, figure_high))
    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/R2/def_strategy_counter.pkl", "rb")
        def_strategy = pickle.load(the_file)
        the_file.close()

        strategy_counter = np.zeros(strategy_number)
        for key in def_strategy.keys():
            for strat_list in def_strategy[key]:
                for strat_id in strat_list:
                    strategy_counter[strat_id] += 1

        strategy_probability = strategy_counter / np.sum(strategy_counter)

        ax.bar(x + notation * width / bar_group_number, strategy_probability, width / bar_group_number,
               label=schemes[schemes_index], hatch=patterns[schemes_index])
        notation += 1
        # plt.bar(range(1, len(strategy_probability) + 1), strategy_probability, hatch=patterns[schemes_index])
    plt.legend()
    # plt.title(schemes, fontsize=font_size)
    plt.xlabel("Defense Strategy ID", fontsize=font_size)
    plt.ylabel("Probability", fontsize=font_size / 1.5)
    plt.xticks(range(1, len(strategy_probability) + 1), fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    plt.tight_layout()
    os.makedirs("Figure/All-In-One", exist_ok=True)
    plt.savefig("Figure/All-In-One/def-Strat-prob-distribution_AllInOne.svg", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/def-Strat-prob-distribution_AllInOne.png", dpi=figure_dpi)
    plt.show()


def display_number_of_attacker():
    # schemes = ["DD-IPI", "DD-Random-IPI","DD-ML-IPI", "DD-PI", "DD-Random-PI","DD-ML-PI"]

    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/R8/number_of_att.pkl", "rb")
        att_number_all_result = pickle.load(the_file)
        the_file.close()

        max_length = 0
        for dict_index in range(len(att_number_all_result)):
            if len(att_number_all_result[dict_index]) > max_length:
                max_length = len(att_number_all_result[dict_index])

        summed_att_number_list = np.zeros(max_length)
        denominator_list = np.zeros(max_length)
        for key in att_number_all_result.keys():
            summed_att_number_list[:len(att_number_all_result[key])] += att_number_all_result[key]
            denominator_list[:len(att_number_all_result[key])] += np.ones(len(att_number_all_result[key]))

        averaged_att_number_list = summed_att_number_list / denominator_list

        # display all simulation results
        plt.figure(figsize=(figure_width, figure_high))
        for key in att_number_all_result.keys():
            plt.plot(range(len(att_number_all_result[key])), att_number_all_result[key], color='C0', linestyle=':',
                     linewidth=figure_linewidth / 3)
        # add legend
        plt.plot(0, 0, color='C0', linestyle=':', linewidth=figure_linewidth / 3, label="per simulation")

        # display averaged result
        plt.plot(range(len(averaged_att_number_list)), averaged_att_number_list, color='C3', linewidth=figure_linewidth,
                 label="average")

        # Red: Averaged , Blue: Per Simulation
        plt.legend(prop={"size": legend_size}, ncol=4, bbox_to_anchor=(0, 1, 1, 0), loc='lower left', mode="expand")
        plt.xlabel("# of games", fontsize=font_size)
        plt.ylabel("Number of attacker", fontsize=font_size)
        plt.xticks(fontsize=axis_size)
        plt.yticks(fontsize=axis_size)
        plt.tight_layout()
        os.makedirs("Figure/" + schemes[schemes_index], exist_ok=True)
        plt.savefig("Figure/" + schemes[schemes_index] + "/number-of-attacker.svg", dpi=figure_dpi)
        plt.savefig("Figure/" + schemes[schemes_index] + "/number-of-attacker.png", dpi=figure_dpi)
        plt.show()


def display_attacker_CKC():
    # schemes = ["DD-IPI", "DD-Random-IPI","DD-ML-IPI", "DD-PI", "DD-Random-PI","DD-ML-PI"]


    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/R9/att_CKC.pkl", "rb")
        att_CKC_all_result = pickle.load(the_file)
        the_file.close()

        max_length = 0
        for key in att_CKC_all_result.keys():
            if len(att_CKC_all_result[key]) > max_length:
                max_length = len(att_CKC_all_result[key])
        att_CKC_counter = np.zeros((6, max_length))  # 6 is the number of CKC stages.
        for key in att_CKC_all_result.keys():
            for CKC_id in range(6):
                att_CKC_counter[CKC_id, :len(att_CKC_all_result[key])] += [k.count(CKC_id) for k in
                                                                           att_CKC_all_result[key]]

        CKC_sum_in_each_game = [sum(k) for k in att_CKC_counter.T]

        # By Number
        plt.figure(figsize=(figure_width, figure_high))
        for index in range(6):
            plt.plot(range(max_length), att_CKC_counter[index], label=f"CKC #{index}")
        plt.legend(prop={"size": legend_size}, ncol=4, bbox_to_anchor=(0, 1, 1, 0), loc='lower left', mode="expand")
        plt.xlabel("# of games", fontsize=font_size)
        plt.ylabel("Number of Attackers in CKC", fontsize=font_size / 1.5)
        plt.xticks(fontsize=axis_size)
        plt.yticks(fontsize=axis_size)
        plt.tight_layout()
        os.makedirs("Figure/" + schemes[schemes_index], exist_ok=True)
        plt.savefig("Figure/" + schemes[schemes_index] + "/att-CKC-in-Number.svg", dpi=figure_dpi)
        plt.savefig("Figure/" + schemes[schemes_index] + "/att-CKC-in-Number.png", dpi=figure_dpi)
        plt.show()

        # By Percentage
        plt.figure(figsize=(figure_width, figure_high))
        percentage_style_att_CKC_counter = att_CKC_counter / CKC_sum_in_each_game
        for CKC_id in range(6):
            plt.plot(range(max_length), percentage_style_att_CKC_counter[CKC_id], label=f"CKC #{CKC_id}")
        plt.legend(prop={"size": legend_size}, ncol=4, bbox_to_anchor=(0, 1, 1, 0), loc='lower left', mode="expand")
        plt.xlabel("# of games", fontsize=font_size)
        plt.ylabel("Percentage of Attacker in CKC", fontsize=font_size / 1.5)
        plt.xticks(fontsize=axis_size)
        plt.yticks(fontsize=axis_size)
        plt.tight_layout()
        plt.savefig("Figure/" + schemes[schemes_index] + "/att-CKC-in-Percentage.svg", dpi=figure_dpi)
        plt.savefig("Figure/" + schemes[schemes_index] + "/att-CKC-in-Percentage.png", dpi=figure_dpi)
        plt.show()


def display_eviction_record():
    # schemes = ["DD-IPI", "DD-Random-IPI","DD-ML-IPI", "DD-PI", "DD-Random-PI","DD-ML-PI"]

    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/R_self_2/evict_reason.pkl", "rb")
        evict_reason_all_result = pickle.load(the_file)
        the_file.close()

        number_of_bar = 5
        bar_label = ('Honeypot', 'DS 4', 'Attack Itself', 'IDS', 'AS8 Success')
        evict_sum = np.zeros(number_of_bar)
        for key in evict_reason_all_result.keys():
            evict_sum += evict_reason_all_result[key]

        # cumulative result to single result
        evict_sum[3] = max(evict_sum[3] - evict_sum[2], 0)  # meanless to have negative
        evict_sum[2] = max(evict_sum[2] - evict_sum[1], 0)

        plt.figure(figsize=(figure_width, figure_high))
        plt.bar(range(number_of_bar), evict_sum, hatch=patterns[schemes_index])
        # plt.set_xticklabels(["1", "2", "3"])
        plt.xticks(np.arange(number_of_bar), bar_label)
        # plt.xticks(("1", "2", "3"), fontsize=font_size)
        plt.ylabel("Number of Attacker Evicted", fontsize=font_size / 1.5)
        os.makedirs("Figure/" + schemes[schemes_index], exist_ok=True)
        plt.savefig("Figure/" + schemes[schemes_index] + "/att-evict-reason.svg", dpi=figure_dpi)
        plt.savefig("Figure/" + schemes[schemes_index] + "/att-evict-reason.png", dpi=figure_dpi)
        plt.show()


def display_per_Strategy_HEU():
    # schemes = ["DD-IPI", "DD-Random-IPI","DD-ML-IPI", "DD-PI", "DD-Random-PI","DD-ML-PI"]


    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/R_self_5/AHEU_for_all_strategy_DD_IPI.pkl", "rb")
        AHEU_per_Strategy_all_result = pickle.load(the_file)
        the_file.close()

        max_length = 0
        for key in AHEU_per_Strategy_all_result.keys():
            if len(AHEU_per_Strategy_all_result[key]) > max_length:
                max_length = len(AHEU_per_Strategy_all_result[key])

        # AHEU
        AHEU_value_per_Strategy = np.zeros((strategy_number, max_length))
        AHEU_counter_per_Strategy = np.zeros((strategy_number, max_length))
        for key in AHEU_per_Strategy_all_result.keys():
            game_counter = 0
            for AHEU_per_game in AHEU_per_Strategy_all_result[key]:
                for attacker_ID in AHEU_per_game:
                    AHEU_value_per_Strategy[:, game_counter] += AHEU_per_game[attacker_ID]
                    AHEU_counter_per_Strategy[:, game_counter] += np.ones(strategy_number)
                game_counter += 1

        AHEU_value_per_Strategy_averaged = AHEU_value_per_Strategy / AHEU_counter_per_Strategy
        plt.figure(figsize=(figure_width, figure_high))
        for strategy_id in range(strategy_number):
            plt.plot(range(max_length), AHEU_value_per_Strategy_averaged[strategy_id], label=f"Stra {strategy_id + 1}")
        plt.legend(prop={"size": legend_size}, ncol=4, bbox_to_anchor=(0, 1, 1, 0), loc='lower left', mode="expand")
        plt.xlabel("# of games", fontsize=font_size)
        plt.ylabel("C-AHEU Value", fontsize=font_size / 1.5)
        plt.xticks(fontsize=axis_size)
        plt.yticks(fontsize=axis_size)
        plt.tight_layout()
        os.makedirs("Figure/" + schemes[schemes_index], exist_ok=True)
        plt.savefig("Figure/" + schemes[schemes_index] + "/AHEU-per_stratety.svg", dpi=figure_dpi)
        plt.savefig("Figure/" + schemes[schemes_index] + "/AHEU-per_stratety.png", dpi=figure_dpi)
        plt.show()

    # DHEU
    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/R_self_5/DHEU_for_all_strategy_DD_IPI.pkl", "rb")
        DHEU_per_Strategy_all_result = pickle.load(the_file)
        the_file.close()

        max_length = 0
        for key in DHEU_per_Strategy_all_result.keys():
            if len(DHEU_per_Strategy_all_result[key]) > max_length:
                max_length = len(DHEU_per_Strategy_all_result[key])

        DHEU_value_per_Strategy = np.zeros((strategy_number, max_length))
        DHEU_counter_per_Strategy = np.zeros((strategy_number, max_length))

        for key in DHEU_per_Strategy_all_result.keys():
            game_counter = 0
            for DHEU_per_game in DHEU_per_Strategy_all_result[key]:
                DHEU_value_per_Strategy[:, game_counter] += DHEU_per_game
                DHEU_counter_per_Strategy[:, game_counter] += np.ones(strategy_number)
                game_counter += 1

        DHEU_value_per_Strategy_average = DHEU_value_per_Strategy / DHEU_counter_per_Strategy

        plt.figure(figsize=(figure_width, figure_high))
        for strategy_id in range(strategy_number):
            plt.plot(range(max_length), DHEU_value_per_Strategy_average[strategy_id], label=f"Stra {strategy_id + 1}")
        plt.legend(prop={"size": legend_size}, ncol=4, bbox_to_anchor=(0, 1, 1, 0), loc='lower left', mode="expand")
        plt.xlabel("# of games", fontsize=font_size)
        plt.ylabel("C-DHEU Value", fontsize=font_size / 1.5)
        plt.xticks(fontsize=axis_size)
        plt.yticks(fontsize=axis_size)
        plt.tight_layout()
        os.makedirs("Figure/" + schemes[schemes_index], exist_ok=True)
        plt.savefig("Figure/" + schemes[schemes_index] + "/DHEU-per_stratety.svg", dpi=figure_dpi)
        plt.savefig("Figure/" + schemes[schemes_index] + "/DHEU-per_stratety.png", dpi=figure_dpi)
        plt.show()


def display_compromise_probability():
    # schemes = ["DD-IPI", "DD-Random-IPI","DD-ML-IPI", "DD-PI", "DD-Random-PI","DD-ML-PI"]

    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/R_self_5/compromise_probability_all_result.pkl", "rb")
        compromised_probability = pickle.load(the_file)
        the_file.close()

        final_mean_list = []
        for key in compromised_probability.keys():
            if compromised_probability[key]:
                final_mean_list.append(np.mean(compromised_probability[key]))

        if final_mean_list:
            print(f"{schemes[schemes_index]}: Probability of attacker compromise the first node is {np.mean(final_mean_list)}.")
        else:
            print(f"{schemes[schemes_index]}: No result for the compromise probability.")

        plt.figure(figsize=(figure_width, figure_high))


def display_inside_attacker_number():
    # schemes = ["DD-IPI", "DD-Random-IPI","DD-ML-IPI", "DD-PI", "DD-Random-PI","DD-ML-PI"]

    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/R_self_5/number_of_inside_attacker_all_result.pkl", "rb")
        number_of_inside_attacker = pickle.load(the_file)
        the_file.close()

        max_length = 0
        for key in number_of_inside_attacker.keys():
            if len(number_of_inside_attacker[key]) > max_length:
                max_length = len(number_of_inside_attacker[key])

        sum_of_total_attacker = np.zeros(max_length)
        sum_of_insider_attacker = np.zeros(max_length)
        counter_of_sum = np.zeros(max_length)

        for key in number_of_inside_attacker.keys():
            game_counter = 0
            for result_per_game in number_of_inside_attacker[key]:
                sum_of_total_attacker[game_counter] += result_per_game[0]
                sum_of_insider_attacker[game_counter] += result_per_game[1]
                counter_of_sum[game_counter] += 1
                game_counter += 1

        plt.figure(figsize=(figure_width, figure_high))
        average_of_total_attacker = sum_of_total_attacker / counter_of_sum
        average_of_inside_attacker = sum_of_insider_attacker / counter_of_sum
        plt.plot(range(max_length), average_of_total_attacker, label=f"# of total attacker")
        plt.plot(range(max_length), average_of_inside_attacker, label=f"# of inside attacker")
        plt.legend(prop={"size": legend_size}, ncol=4, bbox_to_anchor=(0, 1, 1, 0), loc='lower left', mode="expand")
        os.makedirs("Figure/" + schemes[schemes_index], exist_ok=True)
        plt.savefig("Figure/" + schemes[schemes_index] + "/inside-attacker-number.svg", dpi=figure_dpi)
        plt.savefig("Figure/" + schemes[schemes_index] + "/inside-attacker-number.png", dpi=figure_dpi)
        plt.show()

def display_inside_attacker_in_one():
    # schemes = ["DD-IPI", "DD-Random-IPI","DD-ML-IPI", "DD-PI", "DD-Random-PI","DD-ML-PI"]
    # schemes = ["DD-IPI", "DD-Random-IPI", "DD-PI"]
    plt.figure(figsize=(figure_width, figure_high))
    color_circle = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/R_self_5/number_of_inside_attacker_all_result.pkl", "rb")
        number_of_inside_attacker = pickle.load(the_file)
        the_file.close()

        max_length = 0
        for key in number_of_inside_attacker.keys():
            if len(number_of_inside_attacker[key]) > max_length:
                max_length = len(number_of_inside_attacker[key])

        sum_of_total_attacker = np.zeros(max_length)
        sum_of_insider_attacker = np.zeros(max_length)
        counter_of_sum = np.zeros(max_length)

        for key in number_of_inside_attacker.keys():
            game_counter = 0
            for result_per_game in number_of_inside_attacker[key]:
                sum_of_total_attacker[game_counter] += result_per_game[0]
                sum_of_insider_attacker[game_counter] += result_per_game[1]
                counter_of_sum[game_counter] += 1
                game_counter += 1


        average_of_total_attacker = sum_of_total_attacker / counter_of_sum
        average_of_inside_attacker = sum_of_insider_attacker / counter_of_sum
        plt.plot(range(max_length), average_of_total_attacker, color=color_circle[schemes_index], label=str(legend_name[schemes_index]))
        plt.plot(range(max_length), average_of_inside_attacker, color=color_circle[schemes_index])
    # plt.legend(prop={"size": legend_size}, ncol=3, bbox_to_anchor=(0, 1, 1, 0), loc='lower left', mode="expand")

    plt.xlabel("# of game", fontsize=font_size)
    plt.ylabel("# of attacker (top is total, bottom is inside)", fontsize=font_size / 1.5)
    plt.xticks(fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    plt.legend(fontsize=legend_size/1.6)
    plt.xlim([0, max_x_length])  # fix x axis range
    plt.tight_layout()
    os.makedirs("Figure/All-In-One", exist_ok=True)
    plt.savefig("Figure/All-In-One/inside_attacker_AllInOne.svg", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/inside_attacker_AllInOne.png", dpi=figure_dpi)
    plt.show()

def display_def_impact():
    # schemes = ["DD-IPI", "DD-Random-IPI","DD-ML-IPI", "DD-PI", "DD-Random-PI","DD-ML-PI"]

    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/R_self_4/def_impact.pkl", "rb")
        def_impact_all_result = pickle.load(the_file)
        the_file.close()

        max_length = 0
        for key in def_impact_all_result.keys():
            if len(def_impact_all_result[key]) > max_length:
                max_length = len(def_impact_all_result[key])

        total_def_impact = np.zeros((max_length, strategy_number))
        def_impact_counter = np.zeros(max_length)

        for key in def_impact_all_result.keys():
            counter = 0
            for def_impact in def_impact_all_result[key]:
                total_def_impact[counter] += def_impact
                def_impact_counter[counter] += 1
                counter += 1

        average_def_impact = np.zeros((max_length, strategy_number))
        for index in range(strategy_number):
            average_def_impact[:,index] = total_def_impact[:,index]/def_impact_counter

        plt.figure(figsize=(figure_width, figure_high))
        for index in range(strategy_number):
            plt.plot(range(max_length), average_def_impact[:,index], label=f"Stra {index + 1}")
        plt.legend(prop={"size": legend_size}, ncol=4, bbox_to_anchor=(0, 1, 1, 0), loc='lower left', mode="expand")
        plt.xlabel("# of games", fontsize=font_size)
        plt.ylabel("defense impact Value", fontsize=font_size / 1.5)
        plt.xticks(fontsize=axis_size)
        plt.yticks(fontsize=axis_size)
        plt.tight_layout()
        os.makedirs("Figure/" + schemes[schemes_index], exist_ok=True)
        plt.savefig("Figure/" + schemes[schemes_index] + "/defense-impact-of-stratety.svg", dpi=figure_dpi)
        plt.savefig("Figure/" + schemes[schemes_index] + "/defense-impact-of-stratety.png", dpi=figure_dpi)
        plt.show()

def display_uncertainty():
    # attacker uncertainty average result
    # schemes =  ["DD-IPI", "DD-Random-IPI", "DD-PI"]#, "DD-PI", "DD-Random-PI", "DD-ML-PI"]

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
                    if len(att_uncertainty_history[key][0]) > 0:
                        # sum_on_index += att_uncertainty_history[key][0]
                        sum_on_index += np.sum(att_uncertainty_history[key][0])/len(att_uncertainty_history[key][0])
                        att_uncertainty_history[key].pop(0)
                        number_on_index += 1
            average_att_uncertainty.append(sum_on_index / number_on_index)

        x_values = range(len(average_att_uncertainty))
        y_values = average_att_uncertainty
        plt.plot(x_values, y_values, linestyle=all_linestyle[schemes_index], label=legend_name[schemes_index],
                 linewidth=figure_linewidth, marker=marker_list[schemes_index], markevery=50, markersize=marker_size)
    plt.legend(prop={"size":legend_size/1.2}, ncol=4, bbox_to_anchor=(-0.18, 1, 1.2, 0), # ncol=2, bbox_to_anchor=(0, 1, 1, 0),
                  loc='lower left', fontsize='large',mode="expand")
    # plt.legend(prop={"size":legend_size}, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.xlabel("number of games", fontsize=font_size)
    plt.ylabel("Att-Uncertainty", fontsize=font_size)
    plt.xticks(fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    plt.xlim([0, max_x_length])  # fix x axis range
    plt.tight_layout()
    os.makedirs("Figure/All-In-One", exist_ok=True)
    plt.savefig("Figure/All-In-One/att-uncertain-AllInOne.svg", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/att-uncertain-AllInOne.png", dpi=figure_dpi)
    plt.show()

    # defender uncertainty average result

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
                    # sum_on_index += np.sum(def_uncertainty_history[key][0])/len(def_uncertainty_history[key][0])
                    def_uncertainty_history[key].pop(0)
                    number_on_index += 1
            average_def_uncertainty.append(sum_on_index / number_on_index)

        x_values = range(len(average_def_uncertainty))
        y_values = average_def_uncertainty
        plt.plot(x_values, y_values, linestyle=all_linestyle[schemes_index], label=legend_name[schemes_index],
                 linewidth=figure_linewidth, marker=marker_list[schemes_index], markevery=50, markersize=marker_size)
    plt.legend(prop={"size":legend_size/1.2}, ncol=4, bbox_to_anchor=(-0.18, 1, 1.2, 0), # ncol=2, bbox_to_anchor=(0, 1, 1, 0),
                  loc='lower left', fontsize='large',mode="expand")
    plt.xlabel("number of games", fontsize=font_size)
    plt.ylabel("Def-Uncertainty", fontsize=font_size)
    plt.xticks(fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    plt.xlim([0, max_x_length])  # fix x axis range
    plt.tight_layout()

    os.makedirs("Figure/All-In-One", exist_ok=True)
    plt.savefig("Figure/All-In-One/def-uncertain-AllInOne.svg", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/def-uncertain-AllInOne.png", dpi=figure_dpi)
    plt.show()

def display_average_uncertainty():
    # attacker uncertainty average result
    # schemes =  ["DD-IPI", "DD-Random-IPI", "DD-PI"]#, "DD-PI", "DD-Random-PI", "DD-ML-PI"]

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
                    if len(att_uncertainty_history[key][0]) > 0:
                        # sum_on_index += att_uncertainty_history[key][0]
                        sum_on_index += np.sum(att_uncertainty_history[key][0])/len(att_uncertainty_history[key][0])
                        att_uncertainty_history[key].pop(0)
                        number_on_index += 1
            average_att_uncertainty.append(sum_on_index / number_on_index)

        x_values = range(len(average_att_uncertainty))
        y_values = average_att_uncertainty
        plt.bar(schemes_index, np.mean(y_values), hatch=patterns[schemes_index], label=schemes[schemes_index])

    # plt.legend(prop={"size":legend_size}, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.xlabel("Schemes", fontsize=font_size)
    plt.ylabel("Average Att-Uncertainty", fontsize=font_size)
    plt.xticks(range(len(schemes) + 1), [textwrap.fill(label, 7) for label in schemes], fontsize=0.6*axis_size)
    plt.yticks(fontsize=axis_size)
    plt.tight_layout()
    os.makedirs("Figure/All-In-One", exist_ok=True)
    plt.savefig("Figure/All-In-One/att_average_uncertain.svg", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/att_average_uncertain.png", dpi=figure_dpi)
    plt.show()

    # defender uncertainty average result

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
                    # sum_on_index += np.sum(def_uncertainty_history[key][0])/len(def_uncertainty_history[key][0])
                    def_uncertainty_history[key].pop(0)
                    number_on_index += 1
            average_def_uncertainty.append(sum_on_index / number_on_index)

        x_values = range(len(average_def_uncertainty))
        y_values = average_def_uncertainty
        plt.bar(schemes_index, np.mean(y_values), hatch=patterns[schemes_index], label=schemes[schemes_index])

    plt.xlabel("Schemes", fontsize=font_size)
    plt.ylabel("Average Def-Uncertainty", fontsize=font_size)
    plt.xticks(range(len(schemes) + 1), [textwrap.fill(label, 7) for label in legend_name], fontsize=0.6*axis_size)
    plt.yticks(fontsize=axis_size)
    plt.tight_layout()

    os.makedirs("Figure/All-In-One", exist_ok=True)
    plt.savefig("Figure/All-In-One/def_average_uncertain.svg", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/def_average_uncertain.png", dpi=figure_dpi)
    plt.show()

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

def display_SysFail_in_one():
    fig, ax = plt.subplots(figsize=(figure_width, figure_high))

    # shift_value = [- width / 2 - width, - width / 2, + width / 2, width / 2 + width]
    set_width = 0.9
    width = set_width/len(schemes)
    shift_value = (np.arange(len(schemes))- (len(schemes)-1)/2)/len(schemes) * set_width

    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/R_self_3/system_fail.pkl", "rb")
        SysFail_reason = pickle.load(the_file)
        the_file.close()

        print(SysFail_reason)
        y_values = SysFail_reason
        temp_x = np.arange(len(y_values))
        rects = ax.bar(temp_x + shift_value[schemes_index], y_values, width, label=schemes[schemes_index], hatch=patterns[schemes_index])
        autolabel(rects, ax)

    x_values = ["All node Evicted", "SF condition 1", "SF condition 2"]
    ax.legend(prop={"size": legend_size})
    ax.set_xticks(temp_x)
    ax.set_xticklabels(x_values)
    ax.set_xlabel("Reasons for System Failure", fontsize=font_size)
    ax.set_ylabel("number of simulation", fontsize=font_size)
    plt.tight_layout()
    os.makedirs("Figure/All-In-One", exist_ok=True)
    plt.savefig("Figure/All-In-One/SysFail_all_in_one_bar.svg", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/SysFail_all_in_one_bar.png", dpi=figure_dpi)
    plt.show()

def display_impact():
    # Attacker impact
    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/R_self_4/att_impact.pkl", "rb")
        att_impact_all_result = pickle.load(the_file)
        the_file.close()

        plt.figure(figsize=(figure_width, figure_high))
        max_length = 0
        for key in att_impact_all_result.keys():
            if max_length < len(att_impact_all_result[key]):
                max_length = len(att_impact_all_result[key])

        average_impact = []
        for index in range(max_length):
            sum_on_index = 0
            number_on_index = 0
            att_impact = np.zeros(8)
            for key in att_impact_all_result.keys():
                if len(att_impact_all_result[key]) > 0:
                    att_impact = np.add(att_impact, att_impact_all_result[key][0])
                    att_impact_all_result[key] = np.delete(att_impact_all_result[key], 0, 0)
                    number_on_index += 1
            average_impact.append((att_impact / number_on_index).tolist())
        average_impact = np.array(average_impact)

        for index in range(8):
            plt.plot(range(max_length), average_impact[:, index], linestyle=all_linestyle[index], label=f"Stra {index + 1}")
        plt.legend(prop={"size": legend_size})
        plt.title(schemes[schemes_index], fontsize=font_size)
        plt.xlabel("number of games", fontsize=font_size)
        plt.ylabel(f"Att's impact in {schemes[schemes_index]}", fontsize=font_size)
        os.makedirs("Figure/" + schemes[schemes_index], exist_ok=True)
        plt.savefig("Figure/" + schemes[schemes_index] + "/attack-impact-of-stratety.svg", dpi=figure_dpi)
        plt.savefig("Figure/" + schemes[schemes_index] + "/attack-impact-of-stratety.png", dpi=figure_dpi)
        plt.show()

def display_average_TPR():
    plt.figure(figsize=(figure_width, figure_high + 0.75))
    error = []
    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/R4/TPR.pkl", "rb")
        TPR_history = pickle.load(the_file)
        the_file.close()

        TPR_list = []
        for key in TPR_history.keys():
            TPR_list += TPR_history[0]
        mean_TPR = np.mean(TPR_list)

        print(f"{schemes[schemes_index]}: {mean_TPR}")
        plt.bar(schemes_index, mean_TPR, label=schemes[schemes_index], yerr = np.std(TPR_list), capsize=8, hatch=patterns[schemes_index])

    plt.xlabel("Schemes", fontsize=font_size)
    plt.ylabel("TPR", fontsize=font_size)
    plt.xticks(np.arange(len(schemes)), [textwrap.fill(label, 7) for label in schemes], fontsize=0.6*axis_size)
    plt.yticks(fontsize=axis_size)
    plt.ylim(0.89, 0.93)
    plt.tight_layout()
    os.makedirs("Figure/All-In-One", exist_ok=True)
    plt.savefig("Figure/All-In-One/average_TPR_AllInOne.svg", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/average_TPR_AllInOne.png", dpi=figure_dpi)
    plt.show()

def display_TPR():
    plt.figure(figsize=(figure_width, figure_high))
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

        plt.plot(x_values, y_values, linestyle=all_linestyle[schemes_index], label=legend_name[schemes_index],
                 linewidth=figure_linewidth, marker=marker_list[schemes_index], markevery=20, markersize=marker_size)
    plt.legend(prop={"size": legend_size/1.2},
               ncol=4,
               bbox_to_anchor=(-0.25, 1.01, 1.25, 0),
               loc='lower left',
               mode="expand")
    plt.xlabel("number of games", fontsize=font_size)
    plt.ylabel("TPR", fontsize=font_size)
    plt.xticks(fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    plt.xlim([0, max_x_length])  # fix x axis range
    plt.ylim([0.899, 0.935])  # fix x axis range
    plt.tight_layout()
    os.makedirs("Figure/All-In-One", exist_ok=True)
    plt.savefig("Figure/All-In-One/TPR_AllInOne.svg", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/TPR_AllInOne.png", dpi=figure_dpi)
    plt.show()


def display_FPR():
    plt.figure(figsize=(figure_width, figure_high))
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

        plt.plot(x_values, y_values, linestyle=all_linestyle[schemes_index], label=legend_name[schemes_index],
                 linewidth=figure_linewidth, marker=marker_list[schemes_index], markevery=20, markersize=marker_size)
    plt.legend(prop={"size": legend_size/1.2},
               ncol=4,
               bbox_to_anchor=(-0.25, 1.01, 1.25, 0),
               loc='lower left',
               mode="expand")
    plt.xlabel("number of games", fontsize=font_size)
    plt.ylabel("FPR", fontsize=font_size)
    plt.xticks(fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    plt.xlim([0, max_x_length])  # fix x axis range
    plt.ylim([0.065, 0.101])  # fix x axis range
    plt.tight_layout()
    os.makedirs("Figure/All-In-One", exist_ok=True)
    plt.savefig("Figure/All-In-One/FPR_AllInOne.svg", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/FPR_AllInOne.png", dpi=figure_dpi)
    plt.show()

def display_average_cost():
    # attacker cost
    plt.figure(figsize=(figure_width, figure_high + 0.75))
    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/R6/att_cost.pkl", "rb")
        att_cost_all_result = pickle.load(the_file)
        the_file.close()

        cost_list = []
        for key in att_cost_all_result.keys():
            for att_cost in att_cost_all_result[key]:
                cost_list = np.concatenate((cost_list, att_cost))

        plt.bar(schemes_index, np.mean(cost_list), yerr=np.std(cost_list), capsize=10, hatch=patterns[schemes_index])
        plt.text(schemes_index + 0.03, np.mean(cost_list) + 0.01, round(np.mean(cost_list), 2))

    plt.xlabel("Schemes", fontsize=font_size)
    plt.ylabel("Attack cost", fontsize=font_size)
    plt.xticks(np.arange(len(schemes)), [textwrap.fill(label, 7) for label in schemes],fontsize=0.6*axis_size)
    plt.yticks(fontsize=axis_size)
    plt.tight_layout()
    os.makedirs("Figure/All-In-One", exist_ok=True)
    plt.savefig("Figure/All-In-One/att_average_cost.svg", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/att_average_cost.png", dpi=figure_dpi)
    plt.show()

    # defender cost
    plt.figure(figsize=(figure_width, figure_high + 0.75))
    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/R6/def_cost.pkl", "rb")
        def_cost_all_result = pickle.load(the_file)
        the_file.close()

        cost_list = []
        for key in def_cost_all_result.keys():
            for def_cost in def_cost_all_result[key]:
                cost_list = np.concatenate((cost_list, def_cost))

        plt.bar(schemes_index, np.mean(cost_list), yerr=np.std(cost_list), capsize=10, hatch=patterns[schemes_index])
        plt.text(schemes_index + 0.03, np.mean(cost_list) + 0.01, round(np.mean(cost_list), 2))

    plt.xlabel("Schemes", fontsize=font_size)
    plt.ylabel("Defence cost", fontsize=font_size)
    plt.xticks(np.arange(len(schemes)), [textwrap.fill(label, 7) for label in schemes], fontsize=0.6 * axis_size)
    plt.yticks(fontsize=axis_size)
    plt.tight_layout()
    os.makedirs("Figure/All-In-One", exist_ok=True)
    plt.savefig("Figure/All-In-One/def_average_cost.svg", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/def_average_cost.png", dpi=figure_dpi)
    plt.show()

def display_cost():
    # attacker
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
                    sum_on_index += np.mean(att_cost_all_result[key][0])
                    att_cost_all_result[key].pop(0)
                    number_on_index += 1
            average_att_cost.append(sum_on_index / number_on_index)

        x_values = range(len(average_att_cost))
        y_values = average_att_cost

        plt.plot(x_values[1:], y_values[1:], linestyle=all_linestyle[schemes_index], label=legend_name[schemes_index])
    plt.legend(prop={"size": legend_size/1.2},
               ncol=4,
               bbox_to_anchor=(-0.18, 1, 1.2, 0),
               loc='lower left',
               mode="expand")
    plt.xlabel("number of games", fontsize=font_size)
    plt.ylabel("Attack cost", fontsize=font_size)
    plt.xticks(fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    plt.xlim([0, max_x_length])  # fix x axis range
    plt.tight_layout()
    os.makedirs("Figure/All-In-One", exist_ok=True)
    plt.savefig("Figure/All-In-One/att_cost.svg", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/att_cost.png", dpi=figure_dpi)
    plt.show()

    # defender
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
                    sum_on_index += np.mean(def_cost_all_result[key][0])
                    def_cost_all_result[key].pop(0)
                    number_on_index += 1
            average_def_cost.append(sum_on_index / number_on_index)

        x_values = range(len(average_def_cost))
        y_values = average_def_cost

        plt.plot(x_values[1:], y_values[1:], linestyle=all_linestyle[schemes_index], label=legend_name[schemes_index])
    plt.legend(prop={"size": legend_size/1.2},
               ncol=4,
               bbox_to_anchor=(-0.18, 1, 1.2, 0),
               loc='lower left',
               mode="expand")
    plt.xlabel("number of games", fontsize=font_size)
    plt.ylabel("Defence cost", fontsize=font_size)
    plt.xticks(fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    plt.xlim([0, max_x_length])  # fix x axis range
    plt.tight_layout()
    os.makedirs("Figure/All-In-One", exist_ok=True)
    plt.savefig("Figure/All-In-One/def_cost.svg", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/def_cost.png", dpi=figure_dpi)
    plt.show()

def display_hitting_prob(schemes, legend_name):
    max_x_axis = 125
    plt.figure(figsize=(figure_width, figure_high))
    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/R_self_5/hitting_probability.pkl", "rb")
        hitting_prob = pickle.load(the_file)
        the_file.close()

        max_length = 0
        for key in hitting_prob.keys():
            if max_length < len(hitting_prob[key]):
                max_length = len(hitting_prob[key])

        if max_length > max_x_axis:
            max_length = max_x_axis

        print(hitting_prob)
        print(max_length)
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

        print(hit_prob)
        x_values = range(len(hit_prob))
        y_values = hit_prob
        plt.plot(x_values, y_values, linestyle=all_linestyle[schemes_index], label=legend_name[schemes_index],
                     linewidth=figure_linewidth, marker=marker_list[schemes_index], markevery=50, markersize=marker_size)
    plt.legend(prop={"size": legend_size},
               ncol=1,
               loc='upper left')
    # bbox_to_anchor = (-0.25, 1.01, 1.25, 0),
    plt.xlabel("number of games", fontsize=font_size)
    plt.ylabel("HNE hitting ratio", fontsize=font_size)
    plt.xticks(fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    plt.tight_layout()
    os.makedirs("Figure/All-In-One", exist_ok=True)
    plt.savefig("Figure/All-In-One/Hitting_Prob_AllInOne.svg", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/Hitting_Prob_AllInOne.png", dpi=figure_dpi)
    plt.show()

def display_legend():
    fig = plt.figure()
    figlegend = plt.figure(figsize=(17.8, 0.5))
    ax = fig.add_subplot(111)


    lines = []
    for schemes_index in range(len(schemes)):
        line, = ax.plot([1, 2, 3], linestyle=all_linestyle[schemes_index], markersize=marker_size/1.5, marker=marker_list[schemes_index], label=legend_name[schemes_index])
        lines.append(line)
    figlegend.legend(handles=lines,prop={"size": legend_size}, ncol=len(legend_name))

    fig.show()
    figlegend.show()
    os.makedirs("Figure/All-In-One", exist_ok=True)
    figlegend.savefig("Figure/All-In-One/legend.svg", dpi=figure_dpi)
    figlegend.savefig("Figure/All-In-One/legend.eps", dpi=figure_dpi)
    figlegend.savefig("Figure/All-In-One/legend.png", dpi=figure_dpi)

def display_MTTSF_varying_VUB():
    plt.figure(figsize=(figure_width, figure_high))
    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/varying_VUB/Vul_Range.pkl", "rb")
        varying_range = pickle.load(the_file)
        the_file = open("data/" + schemes[schemes_index] + "/varying_VUB/MTTSF.pkl", "rb")
        MTTSF = pickle.load(the_file)

        y_axis = np.zeros(len(MTTSF))
        error = np.zeros(len(MTTSF))
        for varying_key in MTTSF.keys():
            y_axis[varying_key] = np.mean(list(MTTSF[varying_key].values()))
            error[varying_key] = np.std(list(MTTSF[varying_key].values()))

        plt.plot(list(MTTSF.keys()), y_axis, linestyle=all_linestyle[schemes_index], linewidth=figure_linewidth, markersize=marker_size, marker=marker_list[schemes_index], label=schemes[schemes_index])
        # plt.errorbar(list(MTTSF.keys()), y_axis, yerr=error, capsize=10)

    if use_legend:
        plt.legend(prop={"size": legend_size},
                   ncol=4,
                   bbox_to_anchor=(-0.17, 1, 1.2, 0),
                   loc='lower left',
                   mode="expand")
    plt.xticks(list(MTTSF.keys()), varying_range[0],fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    plt.xlabel("Vulnerability Upper Bound", fontsize=font_size)
    plt.ylabel("MTTSF", fontsize=font_size)
    plt.tight_layout()
    os.makedirs("Figure/All-In-One/varying_VUB", exist_ok=True)
    plt.savefig("Figure/All-In-One/varying_VUB/MTTSF.svg", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/varying_VUB/MTTSF.eps", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/varying_VUB/MTTSF.png", dpi=figure_dpi)
    plt.show()

def display_cost_varying_VUB():
    # attacker cost
    plt.figure(figsize=(figure_width, figure_high + 0.75))
    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/varying_VUB/Vul_Range.pkl", "rb")
        varying_range = pickle.load(the_file)
        the_file = open("data/" + schemes[schemes_index] + "/varying_VUB/att_cost.pkl", "rb")
        att_cost = pickle.load(the_file)
        the_file.close()

        y_axis = np.zeros(len(att_cost))
        error = np.zeros(len(att_cost))
        for varying_key in att_cost.keys():
            cost_list = []
            att_cost_sum = 0
            att_cost_counter = 0
            for sim_key in att_cost[varying_key].keys():
                for cost_per_game in att_cost[varying_key][sim_key]:
                    att_cost_sum += sum(cost_per_game)
                    cost_list += list(cost_per_game)
                    att_cost_counter += len(cost_per_game)
            y_axis[varying_key] = att_cost_sum/att_cost_counter
            error[varying_key] = np.std(cost_list)

        plt.plot(list(att_cost.keys()), y_axis, linestyle=all_linestyle[schemes_index], linewidth=figure_linewidth, markersize=marker_size, marker=marker_list[schemes_index], label=schemes[schemes_index])
        # plt.errorbar(list(att_cost.keys()), y_axis, yerr=error, capsize=10)

    if use_legend:
        plt.legend(prop={"size": legend_size},
                   ncol=4,
                   bbox_to_anchor=(-0.17, 1, 1.2, 0),
                   loc='lower left',
                   mode="expand")
    plt.xticks(list(att_cost.keys()), varying_range[0], fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    plt.xlabel("Vulnerability Upper Bound", fontsize=font_size)
    plt.ylabel("Attack Cost", fontsize=font_size)
    plt.tight_layout()
    os.makedirs("Figure/All-In-One/varying_VUB", exist_ok=True)
    plt.savefig("Figure/All-In-One/varying_VUB/att_cost.svg", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/varying_VUB/att_cost.eps", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/varying_VUB/att_cost.png", dpi=figure_dpi)
    plt.show()


    # defender cost
    plt.figure(figsize=(figure_width, figure_high + 0.75))
    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/varying_VUB/Vul_Range.pkl", "rb")
        varying_range = pickle.load(the_file)
        the_file = open("data/" + schemes[schemes_index] + "/varying_VUB/def_cost.pkl", "rb")
        def_cost = pickle.load(the_file)
        the_file.close()

        y_axis = np.zeros(len(def_cost))
        error = np.zeros(len(att_cost))
        for varying_key in def_cost.keys():
            cost_list = []
            def_cost_sum = 0
            def_cost_counter = 0
            for sim_key in def_cost[varying_key].keys():
                for cost_per_game in def_cost[varying_key][sim_key]:
                    def_cost_sum += sum(cost_per_game)
                    cost_list += list(cost_per_game)
                    def_cost_counter += len(cost_per_game)
            y_axis[varying_key] = def_cost_sum/def_cost_counter
            error[varying_key] = np.std(cost_list)

        plt.plot(list(def_cost.keys()), y_axis, linestyle=all_linestyle[schemes_index], linewidth=figure_linewidth, markersize=marker_size, marker=marker_list[schemes_index], label=schemes[schemes_index])
        # plt.errorbar(list(att_cost.keys()), y_axis, yerr=error, capsize=10)

    if use_legend:
        plt.legend(prop={"size": legend_size},
                   ncol=4,
                   bbox_to_anchor=(-0.17, 1, 1.2, 0),
                   loc='lower left',
                   mode="expand")
    plt.xticks(list(def_cost.keys()), varying_range[0], fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    plt.xlabel("Vulnerability Upper Bound", fontsize=font_size)
    plt.ylabel("Defense Cost", fontsize=font_size)
    plt.tight_layout()
    os.makedirs("Figure/All-In-One/varying_VUB", exist_ok=True)
    plt.savefig("Figure/All-In-One/varying_VUB/def_cost.svg", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/varying_VUB/def_cost.eps", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/varying_VUB/def_cost.png", dpi=figure_dpi)
    plt.show()

def display_HEU_varying_VUB():
    # AHEU
    plt.figure(figsize=(figure_width, figure_high))
    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/varying_VUB/Vul_Range.pkl", "rb")
        varying_range = pickle.load(the_file)
        the_file = open("data/" + schemes[schemes_index] + "/varying_VUB/att_HEU.pkl", "rb")
        att_HEU = pickle.load(the_file)
        the_file.close()

        y_axis = np.zeros(len(att_HEU))
        for varying_key in att_HEU.keys():
            att_HEU_sum = 0
            att_HEU_counter = 0
            for sim_key in att_HEU[varying_key].keys():
                for HEU_per_game in att_HEU[varying_key][sim_key]:
                    att_HEU_sum += sum(HEU_per_game)
                    att_HEU_counter += len(HEU_per_game)
            y_axis[varying_key] = att_HEU_sum / att_HEU_counter

        plt.plot(list(att_HEU.keys()), y_axis, linestyle=all_linestyle[schemes_index], linewidth=figure_linewidth, markersize=marker_size, marker=marker_list[schemes_index], label=schemes[schemes_index])

    if use_legend:
        plt.legend(prop={"size": legend_size},
                   ncol=4,
                   bbox_to_anchor=(-0.17, 1, 1.2, 0),
                   loc='lower left',
                   mode="expand")
    plt.xticks(list(att_HEU.keys()), varying_range[0], fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    plt.xlabel("Vulnerability Upper Bound", fontsize=font_size)
    plt.ylabel("C-AHEU", fontsize=font_size)
    plt.tight_layout()
    os.makedirs("Figure/All-In-One/varying_VUB", exist_ok=True)
    plt.savefig("Figure/All-In-One/varying_VUB/AHEU.svg", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/varying_VUB/AHEU.eps", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/varying_VUB/AHEU.png", dpi=figure_dpi)
    plt.show()

    # DHEU
    plt.figure(figsize=(figure_width, figure_high))
    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/varying_VUB/Vul_Range.pkl", "rb")
        varying_range = pickle.load(the_file)
        the_file = open("data/" + schemes[schemes_index] + "/varying_VUB/def_HEU.pkl", "rb")
        def_HEU = pickle.load(the_file)
        the_file.close()

        y_axis = np.zeros(len(def_HEU))
        for varying_key in def_HEU.keys():
            def_HEU_sum = 0
            def_HEU_counter = 0
            for sim_key in def_HEU[varying_key].keys():
                for HEU_per_game in def_HEU[varying_key][sim_key]:
                    def_HEU_sum += sum(HEU_per_game)
                    def_HEU_counter += len(HEU_per_game)
            y_axis[varying_key] = def_HEU_sum / def_HEU_counter

        plt.plot(list(def_HEU.keys()), y_axis, linestyle=all_linestyle[schemes_index], linewidth=figure_linewidth, markersize=marker_size, marker=marker_list[schemes_index], label=schemes[schemes_index])

    if use_legend:
        plt.legend(prop={"size": legend_size},
                   ncol=4,
                   bbox_to_anchor=(-0.17, 1, 1.2, 0),
                   loc='lower left',
                   mode="expand")
    plt.xticks(list(def_HEU.keys()), varying_range[0], fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    plt.xlabel("Vulnerability Upper Bound", fontsize=font_size)
    plt.ylabel("C-DHEU", fontsize=font_size)
    plt.tight_layout()
    os.makedirs("Figure/All-In-One/varying_VUB", exist_ok=True)
    plt.savefig("Figure/All-In-One/varying_VUB/DHEU.svg", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/varying_VUB/DHEU.eps", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/varying_VUB/DHEU.png", dpi=figure_dpi)
    plt.show()

def display_uncertainty_varying_VUB():
    # attacker uncertainty
    plt.figure(figsize=(figure_width, figure_high))
    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/varying_VUB/Vul_Range.pkl", "rb")
        varying_range = pickle.load(the_file)
        the_file = open("data/" + schemes[schemes_index] + "/varying_VUB/att_uncertainty.pkl", "rb")
        att_uncertain = pickle.load(the_file)
        the_file.close()

        y_axis = np.zeros(len(att_uncertain))
        for varying_key in att_uncertain.keys():
            att_uncertain_sum = 0
            att_uncertain_counter = 0
            for sim_key in att_uncertain[varying_key].keys():
                for uncertain_per_game in att_uncertain[varying_key][sim_key]:
                    att_uncertain_sum += sum(uncertain_per_game)
                    att_uncertain_counter += len(uncertain_per_game)
            y_axis[varying_key] = att_uncertain_sum / att_uncertain_counter

        plt.plot(list(att_uncertain.keys()), y_axis, linestyle=all_linestyle[schemes_index], linewidth=figure_linewidth, markersize=marker_size, marker=marker_list[schemes_index], label=schemes[schemes_index])

    if use_legend:
        plt.legend(prop={"size": legend_size},
                   ncol=4,
                   bbox_to_anchor=(-0.17, 1, 1.2, 0),
                   loc='lower left',
                   mode="expand")
    plt.xticks(list(att_uncertain.keys()), varying_range[0], fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    plt.xlabel("Vulnerability Upper Bound", fontsize=font_size)
    plt.ylabel("Attacker Uncertainty", fontsize=font_size)
    plt.tight_layout()
    os.makedirs("Figure/All-In-One/varying_VUB", exist_ok=True)
    plt.savefig("Figure/All-In-One/varying_VUB/att_uncertain.svg", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/varying_VUB/att_uncertain.eps", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/varying_VUB/att_uncertain.png", dpi=figure_dpi)
    plt.show()

    # Defender Uncertainty
    plt.figure(figsize=(figure_width, figure_high))
    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/varying_VUB/Vul_Range.pkl", "rb")
        varying_range = pickle.load(the_file)
        the_file = open("data/" + schemes[schemes_index] + "/varying_VUB/def_uncertainty.pkl", "rb")
        def_uncertain = pickle.load(the_file)
        the_file.close()

        y_axis = np.zeros(len(def_uncertain))
        for varying_key in def_uncertain.keys():
            def_uncertain_sum = 0
            def_uncertain_counter = 0
            for sim_key in def_uncertain[varying_key].keys():
                def_uncertain_sum += sum(def_uncertain[varying_key][sim_key])
                def_uncertain_counter += len(def_uncertain[varying_key][sim_key])
            y_axis[varying_key] = def_uncertain_sum / def_uncertain_counter

        plt.plot(list(def_uncertain.keys()), y_axis, linestyle=all_linestyle[schemes_index], linewidth=figure_linewidth, markersize=marker_size, marker=marker_list[schemes_index], label=schemes[schemes_index])

    if use_legend:
        plt.legend(prop={"size": legend_size},
                   ncol=4,
                   bbox_to_anchor=(-0.17, 1, 1.2, 0),
                   loc='lower left',
                   mode="expand")
    plt.xticks(list(def_uncertain.keys()), varying_range[0], fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    plt.xlabel("Vulnerability Upper Bound", fontsize=font_size)
    plt.ylabel("Defender Uncertainty", fontsize=font_size)
    plt.tight_layout()
    os.makedirs("Figure/All-In-One/varying_VUB", exist_ok=True)
    plt.savefig("Figure/All-In-One/varying_VUB/def_uncertain.svg", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/varying_VUB/def_uncertain.eps", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/varying_VUB/def_uncertain.png", dpi=figure_dpi)
    plt.show()

def display_FPR_varying_VUB():
    plt.figure(figsize=(figure_width, figure_high))
    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/varying_VUB/Vul_Range.pkl", "rb")
        varying_range = pickle.load(the_file)
        the_file = open("data/" + schemes[schemes_index] + "/varying_VUB/FPR.pkl", "rb")
        FPR = pickle.load(the_file)
        the_file.close()

        y_axis = np.zeros(len(FPR))
        for varying_key in FPR.keys():
            FPR_sum = 0
            FPR_counter = 0
            for sim_key in FPR[varying_key].keys():
                FPR_sum += sum(FPR[varying_key][sim_key])
                FPR_counter += len(FPR[varying_key][sim_key])
            y_axis[varying_key] = FPR_sum / FPR_counter

        plt.plot(list(FPR.keys()), y_axis, linestyle=all_linestyle[schemes_index], linewidth=figure_linewidth, markersize=marker_size, marker=marker_list[schemes_index], label=schemes[schemes_index])

    if use_legend:
        plt.legend(prop={"size": legend_size},
                   ncol=4,
                   bbox_to_anchor=(-0.17, 1, 1.2, 0),
                   loc='lower left',
                   mode="expand")
    plt.xticks(list(FPR.keys()), varying_range[0], fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    plt.xlabel("Vulnerability Upper Bound", fontsize=font_size)
    plt.ylabel("FPR", fontsize=font_size)
    plt.tight_layout()
    os.makedirs("Figure/All-In-One/varying_VUB", exist_ok=True)
    plt.savefig("Figure/All-In-One/varying_VUB/FPR.svg", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/varying_VUB/FPR.eps", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/varying_VUB/FPR.png", dpi=figure_dpi)
    plt.show()

def display_TPR_varying_VUB():
    plt.figure(figsize=(figure_width, figure_high))
    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/varying_VUB/Vul_Range.pkl", "rb")
        varying_range = pickle.load(the_file)
        the_file = open("data/" + schemes[schemes_index] + "/varying_VUB/TPR.pkl", "rb")
        TPR = pickle.load(the_file)
        the_file.close()

        y_axis = np.zeros(len(TPR))
        for varying_key in TPR.keys():
            TPR_sum = 0
            TPR_counter = 0
            for sim_key in TPR[varying_key].keys():
                TPR_sum += sum(TPR[varying_key][sim_key])
                TPR_counter += len(TPR[varying_key][sim_key])
            y_axis[varying_key] = TPR_sum / TPR_counter

        plt.plot(list(TPR.keys()), y_axis, linestyle=all_linestyle[schemes_index], linewidth=figure_linewidth, markersize=marker_size, marker=marker_list[schemes_index], label=schemes[schemes_index])

    if use_legend:
        plt.legend(prop={"size": legend_size},
                   ncol=4,
                   bbox_to_anchor=(-0.17, 1, 1.2, 0),
                   loc='lower left',
                   mode="expand")
    plt.xticks(list(TPR.keys()), varying_range[0], fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    plt.xlabel("Vulnerability Upper Bound", fontsize=font_size)
    plt.ylabel("TPR", fontsize=font_size)
    plt.tight_layout()
    os.makedirs("Figure/All-In-One/varying_VUB", exist_ok=True)
    plt.savefig("Figure/All-In-One/varying_VUB/TPR.svg", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/varying_VUB/TPR.eps", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/varying_VUB/TPR.png", dpi=figure_dpi)
    plt.show()


def display_MTTSF_varying_AAP():
    plt.figure(figsize=(figure_width, figure_high))
    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/varying_AAP/AAP_Range.pkl", "rb")
        varying_range = pickle.load(the_file)
        the_file = open("data/" + schemes[schemes_index] + "/varying_AAP/MTTSF.pkl", "rb")
        MTTSF = pickle.load(the_file)

        y_axis = np.zeros(len(MTTSF))
        error = np.zeros(len(MTTSF))
        for varying_key in MTTSF.keys():
            y_axis[varying_key] = np.mean(list(MTTSF[varying_key].values()))
            error[varying_key] = np.std(list(MTTSF[varying_key].values()))

        plt.plot(list(MTTSF.keys()), y_axis, linestyle=all_linestyle[schemes_index], linewidth=figure_linewidth, markersize=marker_size, marker=marker_list[schemes_index], label=schemes[schemes_index])
        # plt.errorbar(list(MTTSF.keys()), y_axis, yerr=error, capsize=10)

    if use_legend:
        plt.legend(prop={"size": legend_size},
                   ncol=4,
                   bbox_to_anchor=(-0.17, 1, 1.2, 0),
                   loc='lower left',
                   mode="expand")

    plt.xticks(list(MTTSF.keys()), varying_range,fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    plt.xlabel("Attacker Arrival Probability", fontsize=font_size)
    plt.ylabel("MTTSF", fontsize=font_size)
    plt.tight_layout()
    os.makedirs("Figure/All-In-One/varying_AAP", exist_ok=True)
    plt.savefig("Figure/All-In-One/varying_AAP/MTTSF.svg", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/varying_AAP/MTTSF.eps", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/varying_AAP/MTTSF.png", dpi=figure_dpi)
    plt.show()

def display_cost_varying_AAP():
    # attacker cost
    plt.figure(figsize=(figure_width, figure_high + 0.75))
    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/varying_AAP/AAP_Range.pkl", "rb")
        varying_range = pickle.load(the_file)
        the_file = open("data/" + schemes[schemes_index] + "/varying_AAP/att_cost.pkl", "rb")
        att_cost = pickle.load(the_file)
        the_file.close()

        y_axis = np.zeros(len(att_cost))
        error = np.zeros(len(att_cost))
        for varying_key in att_cost.keys():
            cost_list = []
            att_cost_sum = 0
            att_cost_counter = 0
            for sim_key in att_cost[varying_key].keys():
                for cost_per_game in att_cost[varying_key][sim_key]:
                    att_cost_sum += sum(cost_per_game)
                    cost_list += list(cost_per_game)
                    att_cost_counter += len(cost_per_game)
            y_axis[varying_key] = att_cost_sum/att_cost_counter
            error[varying_key] = np.std(cost_list)

        plt.plot(list(att_cost.keys()), y_axis, linestyle=all_linestyle[schemes_index], linewidth=figure_linewidth, markersize=marker_size, marker=marker_list[schemes_index], label=schemes[schemes_index])
        # plt.errorbar(list(att_cost.keys()), y_axis, yerr=error, capsize=10)

    if use_legend:
        plt.legend(prop={"size": legend_size},
                   ncol=4,
                   bbox_to_anchor=(-0.17, 1, 1.2, 0),
                   loc='lower left',
                   mode="expand")
    plt.xticks(list(att_cost.keys()), varying_range, fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    plt.xlabel("Attacker Arrival Probability", fontsize=font_size)
    plt.ylabel("Attack Cost", fontsize=font_size)
    plt.tight_layout()
    os.makedirs("Figure/All-In-One/varying_AAP", exist_ok=True)
    plt.savefig("Figure/All-In-One/varying_AAP/att_cost.svg", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/varying_AAP/att_cost.eps", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/varying_AAP/att_cost.png", dpi=figure_dpi)
    plt.show()


    # defender cost
    plt.figure(figsize=(figure_width, figure_high + 0.75))
    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/varying_AAP/AAP_Range.pkl", "rb")
        varying_range = pickle.load(the_file)
        the_file = open("data/" + schemes[schemes_index] + "/varying_AAP/def_cost.pkl", "rb")
        def_cost = pickle.load(the_file)
        the_file.close()

        y_axis = np.zeros(len(def_cost))
        error = np.zeros(len(att_cost))
        for varying_key in def_cost.keys():
            cost_list = []
            def_cost_sum = 0
            def_cost_counter = 0
            for sim_key in def_cost[varying_key].keys():
                for cost_per_game in def_cost[varying_key][sim_key]:
                    def_cost_sum += sum(cost_per_game)
                    cost_list += list(cost_per_game)
                    def_cost_counter += len(cost_per_game)
            y_axis[varying_key] = def_cost_sum/def_cost_counter
            error[varying_key] = np.std(cost_list)

        plt.plot(list(def_cost.keys()), y_axis, linestyle=all_linestyle[schemes_index], linewidth=figure_linewidth, markersize=marker_size, marker=marker_list[schemes_index], label=schemes[schemes_index])
        # plt.errorbar(list(att_cost.keys()), y_axis, yerr=error, capsize=10)

    if use_legend:
        plt.legend(prop={"size": legend_size},
                   ncol=4,
                   bbox_to_anchor=(-0.17, 1, 1.2, 0),
                   loc='lower left',
                   mode="expand")
    plt.xticks(list(def_cost.keys()), varying_range, fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    plt.xlabel("Attacker Arrival Probability", fontsize=font_size)
    plt.ylabel("Defense Cost", fontsize=font_size)
    plt.tight_layout()
    os.makedirs("Figure/All-In-One/varying", exist_ok=True)
    plt.savefig("Figure/All-In-One/varying_AAP/def_cost.svg", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/varying_AAP/def_cost.eps", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/varying_AAP/def_cost.png", dpi=figure_dpi)
    plt.show()

def display_HEU_varying_AAP():
    # AHEU
    plt.figure(figsize=(figure_width, figure_high))
    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/varying_AAP/AAP_Range.pkl", "rb")
        varying_range = pickle.load(the_file)
        the_file = open("data/" + schemes[schemes_index] + "/varying_AAP/att_HEU.pkl", "rb")
        att_HEU = pickle.load(the_file)
        the_file.close()

        y_axis = np.zeros(len(att_HEU))
        for varying_key in att_HEU.keys():
            att_HEU_sum = 0
            att_HEU_counter = 0
            for sim_key in att_HEU[varying_key].keys():
                for HEU_per_game in att_HEU[varying_key][sim_key]:
                    att_HEU_sum += sum(HEU_per_game)
                    att_HEU_counter += len(HEU_per_game)
            y_axis[varying_key] = att_HEU_sum / att_HEU_counter

        plt.plot(list(att_HEU.keys()), y_axis, linestyle=all_linestyle[schemes_index], linewidth=figure_linewidth, markersize=marker_size, marker=marker_list[schemes_index], label=schemes[schemes_index])

    if use_legend:
        plt.legend(prop={"size": legend_size},
                   ncol=4,
                   bbox_to_anchor=(-0.17, 1, 1.2, 0),
                   loc='lower left',
                   mode="expand")
    plt.xticks(list(att_HEU.keys()), varying_range, fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    plt.xlabel("Attacker Arrival Probability", fontsize=font_size)
    plt.ylabel("C-AHEU", fontsize=font_size)
    plt.tight_layout()
    os.makedirs("Figure/All-In-One/varying_AAP", exist_ok=True)
    plt.savefig("Figure/All-In-One/varying_AAP/AHEU.svg", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/varying_AAP/AHEU.eps", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/varying_AAP/AHEU.png", dpi=figure_dpi)
    plt.show()

    # DHEU
    plt.figure(figsize=(figure_width, figure_high))
    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/varying_AAP/AAP_Range.pkl", "rb")
        varying_range = pickle.load(the_file)
        the_file = open("data/" + schemes[schemes_index] + "/varying_AAP/def_HEU.pkl", "rb")
        def_HEU = pickle.load(the_file)
        the_file.close()

        y_axis = np.zeros(len(def_HEU))
        for varying_key in def_HEU.keys():
            def_HEU_sum = 0
            def_HEU_counter = 0
            for sim_key in def_HEU[varying_key].keys():
                for HEU_per_game in def_HEU[varying_key][sim_key]:
                    def_HEU_sum += sum(HEU_per_game)
                    def_HEU_counter += len(HEU_per_game)
            y_axis[varying_key] = def_HEU_sum / def_HEU_counter

        plt.plot(list(def_HEU.keys()), y_axis, linestyle=all_linestyle[schemes_index], linewidth=figure_linewidth, markersize=marker_size, marker=marker_list[schemes_index], label=schemes[schemes_index])

    if use_legend:
        plt.legend(prop={"size": legend_size},
                   ncol=4,
                   bbox_to_anchor=(-0.17, 1, 1.2, 0),
                   loc='lower left',
                   mode="expand")
    plt.xticks(list(def_HEU.keys()), varying_range, fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    plt.xlabel("Attacker Arrival Probability", fontsize=font_size)
    plt.ylabel("C-DHEU", fontsize=font_size)
    plt.tight_layout()
    os.makedirs("Figure/All-In-One/varying_AAP", exist_ok=True)
    plt.savefig("Figure/All-In-One/varying_AAP/DHEU.svg", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/varying_AAP/DHEU.eps", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/varying_AAP/DHEU.png", dpi=figure_dpi)
    plt.show()

def display_uncertainty_varying_AAP():
    # attacker uncertainty
    plt.figure(figsize=(figure_width, figure_high))
    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/varying_AAP/AAP_Range.pkl", "rb")
        varying_range = pickle.load(the_file)
        the_file = open("data/" + schemes[schemes_index] + "/varying_AAP/att_uncertainty.pkl", "rb")
        att_uncertain = pickle.load(the_file)
        the_file.close()

        y_axis = np.zeros(len(att_uncertain))
        for varying_key in att_uncertain.keys():
            att_uncertain_sum = 0
            att_uncertain_counter = 0
            for sim_key in att_uncertain[varying_key].keys():
                for uncertain_per_game in att_uncertain[varying_key][sim_key]:
                    att_uncertain_sum += sum(uncertain_per_game)
                    att_uncertain_counter += len(uncertain_per_game)
            y_axis[varying_key] = att_uncertain_sum / att_uncertain_counter

        plt.plot(list(att_uncertain.keys()), y_axis, linestyle=all_linestyle[schemes_index], linewidth=figure_linewidth, markersize=marker_size, marker=marker_list[schemes_index], label=schemes[schemes_index])

    if use_legend:
        plt.legend(prop={"size": legend_size},
                   ncol=4,
                   bbox_to_anchor=(-0.17, 1, 1.2, 0),
                   loc='lower left',
                   mode="expand")
    plt.xticks(list(att_uncertain.keys()), varying_range, fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    plt.xlabel("Attacker Arrival Probability", fontsize=font_size)
    plt.ylabel("Attacker Uncertainty", fontsize=font_size)
    plt.tight_layout()
    os.makedirs("Figure/All-In-One/varying_AAP", exist_ok=True)
    plt.savefig("Figure/All-In-One/varying_AAP/att_uncertain.svg", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/varying_AAP/att_uncertain.eps", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/varying_AAP/att_uncertain.png", dpi=figure_dpi)
    plt.show()

    # Defender Uncertainty
    plt.figure(figsize=(figure_width, figure_high))
    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/varying_AAP/AAP_Range.pkl", "rb")
        varying_range = pickle.load(the_file)
        the_file = open("data/" + schemes[schemes_index] + "/varying_AAP/def_uncertainty.pkl", "rb")
        def_uncertain = pickle.load(the_file)
        the_file.close()

        y_axis = np.zeros(len(def_uncertain))
        for varying_key in def_uncertain.keys():
            def_uncertain_sum = 0
            def_uncertain_counter = 0
            for sim_key in def_uncertain[varying_key].keys():
                def_uncertain_sum += sum(def_uncertain[varying_key][sim_key])
                def_uncertain_counter += len(def_uncertain[varying_key][sim_key])
            y_axis[varying_key] = def_uncertain_sum / def_uncertain_counter

        plt.plot(list(def_uncertain.keys()), y_axis, linestyle=all_linestyle[schemes_index], linewidth=figure_linewidth, markersize=marker_size, marker=marker_list[schemes_index], label=schemes[schemes_index])

    if use_legend:
        plt.legend(prop={"size": legend_size},
                   ncol=4,
                   bbox_to_anchor=(-0.17, 1, 1.2, 0),
                   loc='lower left',
                   mode="expand")
    plt.xticks(list(def_uncertain.keys()), varying_range, fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    plt.xlabel("Attacker Arrival Probability", fontsize=font_size)
    plt.ylabel("Defender Uncertainty", fontsize=font_size)
    plt.tight_layout()
    os.makedirs("Figure/All-In-One/varying_AAP", exist_ok=True)
    plt.savefig("Figure/All-In-One/varying_AAP/def_uncertain.svg", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/varying_AAP/def_uncertain.eps", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/varying_AAP/def_uncertain.png", dpi=figure_dpi)
    plt.show()

def display_FPR_varying_AAP():
    plt.figure(figsize=(figure_width, figure_high))
    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/varying_AAP/AAP_Range.pkl", "rb")
        varying_range = pickle.load(the_file)
        the_file = open("data/" + schemes[schemes_index] + "/varying_AAP/FPR.pkl", "rb")
        FPR = pickle.load(the_file)
        the_file.close()

        y_axis = np.zeros(len(FPR))
        for varying_key in FPR.keys():
            FPR_sum = 0
            FPR_counter = 0
            for sim_key in FPR[varying_key].keys():
                FPR_sum += sum(FPR[varying_key][sim_key])
                FPR_counter += len(FPR[varying_key][sim_key])
            y_axis[varying_key] = FPR_sum / FPR_counter

        plt.plot(list(FPR.keys()), y_axis, linestyle=all_linestyle[schemes_index], linewidth=figure_linewidth, markersize=marker_size, marker=marker_list[schemes_index], label=schemes[schemes_index])

    if use_legend:
        plt.legend(prop={"size": legend_size},
                   ncol=4,
                   bbox_to_anchor=(-0.17, 1, 1.2, 0),
                   loc='lower left',
                   mode="expand")
    plt.xticks(list(FPR.keys()), varying_range, fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    plt.xlabel("Attacker Arrival Probability", fontsize=font_size)
    plt.ylabel("FPR", fontsize=font_size)
    plt.tight_layout()
    os.makedirs("Figure/All-In-One/varying_AAP", exist_ok=True)
    plt.savefig("Figure/All-In-One/varying_AAP/FPR.svg", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/varying_AAP/FPR.eps", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/varying_AAP/FPR.png", dpi=figure_dpi)
    plt.show()


def display_TNR_varying_AAP():
    plt.figure(figsize=(figure_width, figure_high))
    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/varying_AAP/AAP_Range.pkl", "rb")
        varying_range = pickle.load(the_file)
        the_file = open("data/" + schemes[schemes_index] + "/varying_AAP/FPR.pkl", "rb")
        FPR = pickle.load(the_file)
        the_file.close()

        y_axis = np.zeros(len(FPR))
        for varying_key in FPR.keys():
            FPR_sum = 0
            FPR_counter = 0
            for sim_key in FPR[varying_key].keys():
                FPR_sum += sum(FPR[varying_key][sim_key])
                FPR_counter += len(FPR[varying_key][sim_key])
            y_axis[varying_key] = 1 - FPR_sum / FPR_counter     # convert FPR to TNR

        plt.plot(list(FPR.keys()), y_axis, linestyle=all_linestyle[schemes_index], linewidth=figure_linewidth, markersize=marker_size, marker=marker_list[schemes_index], label=schemes[schemes_index])

    if use_legend:
        plt.legend(prop={"size": legend_size},
                   ncol=4,
                   bbox_to_anchor=(-0.17, 1, 1.2, 0),
                   loc='lower left',
                   mode="expand")
    plt.xticks(list(FPR.keys()), varying_range, fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    plt.xlabel("Attacker Arrival Probability", fontsize=font_size)
    plt.ylabel("TNR", fontsize=font_size)
    plt.tight_layout()
    os.makedirs("Figure/All-In-One/varying_AAP", exist_ok=True)
    plt.savefig("Figure/All-In-One/varying_AAP/TNR.svg", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/varying_AAP/TNR.eps", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/varying_AAP/TNR.png", dpi=figure_dpi)
    plt.show()


def display_TPR_varying_AAP():
    plt.figure(figsize=(figure_width, figure_high))
    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/varying_AAP/AAP_Range.pkl", "rb")
        varying_range = pickle.load(the_file)
        the_file = open("data/" + schemes[schemes_index] + "/varying_AAP/TPR.pkl", "rb")
        TPR = pickle.load(the_file)
        the_file.close()

        y_axis = np.zeros(len(TPR))
        for varying_key in TPR.keys():
            TPR_sum = 0
            TPR_counter = 0
            for sim_key in TPR[varying_key].keys():
                TPR_sum += sum(TPR[varying_key][sim_key])
                TPR_counter += len(TPR[varying_key][sim_key])
            y_axis[varying_key] = TPR_sum / TPR_counter

        plt.plot(list(TPR.keys()), y_axis, linestyle=all_linestyle[schemes_index], linewidth=figure_linewidth, markersize=marker_size, marker=marker_list[schemes_index], label=schemes[schemes_index])

    if use_legend:
        plt.legend(prop={"size": legend_size},
                   ncol=4,
                   bbox_to_anchor=(-0.17, 1, 1.2, 0),
                   loc='lower left',
                   mode="expand")
    plt.xticks(list(TPR.keys()), varying_range, fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    plt.xlabel("Attacker Arrival Probability", fontsize=font_size)
    plt.ylabel("TPR", fontsize=font_size)
    plt.tight_layout()
    os.makedirs("Figure/All-In-One/varying_AAP", exist_ok=True)
    plt.savefig("Figure/All-In-One/varying_AAP/TPR.svg", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/varying_AAP/TPR.eps", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/varying_AAP/TPR.png", dpi=figure_dpi)
    plt.show()


def display_FNR_varying_AAP():
    plt.figure(figsize=(figure_width, figure_high))
    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/varying_AAP/AAP_Range.pkl", "rb")
        varying_range = pickle.load(the_file)
        the_file = open("data/" + schemes[schemes_index] + "/varying_AAP/TPR.pkl", "rb")
        TPR = pickle.load(the_file)
        the_file.close()

        y_axis = np.zeros(len(TPR))
        for varying_key in TPR.keys():
            TPR_sum = 0
            TPR_counter = 0
            for sim_key in TPR[varying_key].keys():
                TPR_sum += sum(TPR[varying_key][sim_key])
                TPR_counter += len(TPR[varying_key][sim_key])
            y_axis[varying_key] = 1 - TPR_sum / TPR_counter     # convert TPR to FNR

        plt.plot(list(TPR.keys()), y_axis, linestyle=all_linestyle[schemes_index], linewidth=figure_linewidth, markersize=marker_size, marker=marker_list[schemes_index], label=schemes[schemes_index])

    if use_legend:
        plt.legend(prop={"size": legend_size},
                   ncol=4,
                   bbox_to_anchor=(-0.17, 1, 1.2, 0),
                   loc='lower left',
                   mode="expand")
    plt.xticks(list(TPR.keys()), varying_range, fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    plt.xlabel("Attacker Arrival Probability", fontsize=font_size)
    plt.ylabel("FNR", fontsize=font_size)
    plt.tight_layout()
    os.makedirs("Figure/All-In-One/varying_AAP", exist_ok=True)
    plt.savefig("Figure/All-In-One/varying_AAP/FNR.svg", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/varying_AAP/FNR.eps", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/varying_AAP/FNR.png", dpi=figure_dpi)
    plt.show()


# Recall is the same to the TPR
def display_Recall_varying_AAP():
    plt.figure(figsize=(figure_width, figure_high))
    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/varying_AAP/AAP_Range.pkl", "rb")
        varying_range = pickle.load(the_file)
        the_file = open("data/" + schemes[schemes_index] + "/varying_AAP/TPR.pkl", "rb")
        TPR = pickle.load(the_file)
        print("TPR", TPR)
        the_file.close()

        y_axis = np.zeros(len(TPR))
        for varying_key in TPR.keys():
            TPR_sum = 0
            TPR_counter = 0
            for sim_key in TPR[varying_key].keys():
                TPR_sum += sum(TPR[varying_key][sim_key])
                TPR_counter += len(TPR[varying_key][sim_key])
            y_axis[varying_key] = TPR_sum / TPR_counter

        plt.plot(list(TPR.keys()), y_axis, linestyle=all_linestyle[schemes_index], linewidth=figure_linewidth, markersize=marker_size, marker=marker_list[schemes_index], label=schemes[schemes_index])

    if use_legend:
        plt.legend(prop={"size": legend_size},
                   ncol=4,
                   bbox_to_anchor=(-0.17, 1, 1.2, 0),
                   loc='lower left',
                   mode="expand")
    plt.xticks(list(TPR.keys()), varying_range, fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    plt.xlabel("Attacker Arrival Probability", fontsize=font_size)
    plt.ylabel("Recall", fontsize=font_size)
    plt.tight_layout()
    os.makedirs("Figure/All-In-One/varying_AAP", exist_ok=True)
    plt.savefig("Figure/All-In-One/varying_AAP/Recall.svg", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/varying_AAP/Recall.eps", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/varying_AAP/Recall.png", dpi=figure_dpi)
    plt.show()


# Recall is the same to the TPR
def display_precision_varying_AAP():
    plt.figure(figsize=(figure_width, figure_high))
    for schemes_index in range(len(schemes)):
        the_file = open("data/" + schemes[schemes_index] + "/varying_AAP/AAP_Range.pkl", "rb")
        varying_range = pickle.load(the_file)
        the_file = open("data/" + schemes[schemes_index] + "/varying_AAP/TP.pkl", "rb")
        TP = pickle.load(the_file)
        the_file = open("data/" + schemes[schemes_index] + "/varying_AAP/FP.pkl", "rb")
        FP = pickle.load(the_file)
        print("TP", TP)
        print("FP", FP)
        the_file.close()

        y_axis = np.zeros(len(TP))
        for varying_key in TP.keys():
            # get TP
            TP_sum = 0
            TP_counter = 0
            for sim_key in TP[varying_key].keys():
                TP_sum += sum(TP[varying_key][sim_key])
                TP_counter += len(TP[varying_key][sim_key])
            TP_average = TP_sum / TP_counter
            # get FP
            FP_sum = 0
            FP_counter = 0
            for sim_key in FP[varying_key].keys():
                FP_sum += sum(FP[varying_key][sim_key])
                FP_counter += len(FP[varying_key][sim_key])
            FP_average = FP_sum / FP_counter

            y_axis[varying_key] = TP_average / (TP_average + FP_average)

        plt.plot(list(TP.keys()), y_axis, linestyle=all_linestyle[schemes_index], linewidth=figure_linewidth, markersize=marker_size, marker=marker_list[schemes_index], label=schemes[schemes_index])

    if use_legend:
        plt.legend(prop={"size": legend_size},
                   ncol=4,
                   bbox_to_anchor=(-0.17, 1, 1.2, 0),
                   loc='lower left',
                   mode="expand")
    plt.xticks(list(TP.keys()), varying_range, fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    plt.xlabel("Attacker Arrival Probability", fontsize=font_size)
    plt.ylabel("Precision", fontsize=font_size)
    plt.tight_layout()
    os.makedirs("Figure/All-In-One/varying_AAP", exist_ok=True)
    plt.savefig("Figure/All-In-One/varying_AAP/Precision.svg", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/varying_AAP/Precision.eps", dpi=figure_dpi)
    plt.savefig("Figure/All-In-One/varying_AAP/Precision.png", dpi=figure_dpi)
    plt.show()


if __name__ == '__main__':
    # preset values
    all_linestyle = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    patterns = ["|", "\\", "/", "+", "-", ".", "*", "x", "o", "O"]
    font_size = 20# 25
    figure_high = 5 # 6
    figure_width = 7.5
    figure_linewidth = 3
    figure_dpi = 100
    legend_size = 15 #18
    axis_size = 20
    marker_size = 12
    marker_list = ["p", "d", "v", "x", "s", "*", "1", "."]
    strategy_number = 8
    max_x_length = 60
    use_legend = False
    # "legend_name" has to match the "schemes"
    legend_name =   ["HG-DD-IPI",   "G-DD-PI",  "HG-DD-ML-IPI", "G-DD-ML-PI",   "DD-Random", "HG-No-DD-IPI",    "G-No-DD-PI",   "No-DD-Random"]
    schemes     =   ["DD-IPI",      "DD-PI",    "DD-ML-IPI",    "DD-ML-PI",     "DD-Random", "No-DD-IPI",       "No-DD-PI",     "No-DD-Random"]
    # schemes = ["DD-Random","DD-IPI", "DD-PI"]



    os.makedirs("Figure", exist_ok=True)
    # Display
    # display_TTSF()
    # display_TTSF_in_one()
    # display_per_Strategy_HEU()
    # display_strategy_count()
    # display_number_of_attacker()
    # display_attacker_CKC()
    # display_eviction_record()
    # display_compromise_probability()
    # display_inside_attacker_number()
    # display_def_impact()
    # display_impact()
    # display_strategy_prob_distribution()
    #


    # display_TTSF_in_one_bar()
    # display_average_uncertainty()
    # display_HEU_In_One()
    # display_inside_attacker_in_one()
    # display_TPR()
    # display_FPR()
    # display_cost()
    # display_uncertainty()

    # display_average_HEU_In_One()
    # display_SysFail_in_one()
    # display_strategy_prob_distribution_in_one()
    # display_average_TPR()
    # display_average_cost()
    # display_hitting_prob(["DD-Random","DD-IPI", "DD-PI"], ["DD-Random", "HG-DD-IPI", "G-DD-PI"])

    # display_legend()

    # display_MTTSF_varying_VUB()
    # display_cost_varying_VUB()
    # display_HEU_varying_VUB()
    # display_uncertainty_varying_VUB()
    # display_FPR_varying_VUB()
    # display_TPR_varying_VUB()
    #
    # display_MTTSF_varying_AAP()
    # display_cost_varying_AAP()
    # display_HEU_varying_AAP()
    # display_uncertainty_varying_AAP()
    # display_FPR_varying_AAP()
    # display_TNR_varying_AAP()
    # display_TPR_varying_AAP()
    # display_FNR_varying_AAP()
    display_Recall_varying_AAP()
    display_precision_varying_AAP()


    # varying parameter
    # display_TTSF_vary_AttArivalProb()