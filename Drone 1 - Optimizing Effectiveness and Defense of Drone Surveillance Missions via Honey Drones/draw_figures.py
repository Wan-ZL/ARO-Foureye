import os
import pickle
from collections import defaultdict

from matplotlib import pyplot as plt


def reward_training():
    the_file = open("data/reward_train_all_result.pkl", "rb")
    reward_train_all_result = pickle.load(the_file)
    the_file.close()

    plt.figure(figsize=(figure_width, figure_high))
    plt.plot(range(len(reward_train_all_result)), reward_train_all_result)
    plt.xlabel("Episodes", fontsize=font_size)
    plt.ylabel("Accumulated Reward", fontsize=font_size)
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/reward_episode.pdf", dpi=figure_dpi)
    plt.show()
    print(reward_train_all_result)

def mission_condition():
    the_file = open("data/game_succ_condition.pkl", "rb")
    game_succ_condition = pickle.load(the_file)
    the_file.close()

    plt.figure(figsize=(figure_width, figure_high))
    plt.plot(range(len(game_succ_condition)), game_succ_condition)
    plt.xlabel("Episodes", fontsize=font_size)
    plt.ylabel("Mission Result (0 - success, 1 - scan not finish, 2 - timeout, 3 - no MD alive, ", fontsize=font_size)
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/missionComplete_episode.pdf", dpi=figure_dpi)
    plt.show()
    print(game_succ_condition)


def defense_stra_distribution():
    the_file = open("data/defense_strategy_list.pkl", "rb")
    defense_strategy_list = pickle.load(the_file)
    the_file.close()
    freq_dict = [0]*10  # 10 strategies
    for ele in defense_strategy_list:
        freq_dict[ele] += 1
    plt.figure(figsize=(figure_width, figure_high))
    plt.bar(range(len(freq_dict)), freq_dict)
    plt.xlabel("# of times", fontsize=font_size)
    plt.ylabel("Defense strategy id", fontsize=font_size)
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/dsFreq_dsID.pdf", dpi=figure_dpi)
    plt.show()
    print(freq_dict)





def reward_training_A3C():
    the_file = open("data/A3C/reward_train_all_result.pkl", "rb")
    reward_train_all_result = pickle.load(the_file)
    the_file.close()

    plt.figure(figsize=(figure_width, figure_high))
    plt.plot(range(len(reward_train_all_result)), reward_train_all_result)
    plt.xlabel("Episodes", fontsize=font_size)
    plt.ylabel("Accumulated Reward", fontsize=font_size)
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/reward_episode_A3C.pdf", dpi=figure_dpi)
    plt.show()
    print(reward_train_all_result)

def t_step_A3C():
    the_file = open("data/A3C/t_step_all_result.pkl", "rb")
    t_step_all_result = pickle.load(the_file)
    the_file.close()

    plt.figure(figsize=(figure_width, figure_high))
    plt.plot(range(len(t_step_all_result)), t_step_all_result)
    plt.xlabel("Episodes", fontsize=font_size)
    plt.ylabel("t_step (# of round)", fontsize=font_size)
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/t_step_episode_A3C.pdf", dpi=figure_dpi)
    plt.show()
    print(t_step_all_result)

def defense_stra_distribution_A3C():
    the_file = open("data/A3C/def_action_all_result.pkl", "rb")
    global_def_action = pickle.load(the_file)
    the_file.close()
    plt.figure(figsize=(figure_width, figure_high))
    plt.bar(range(len(global_def_action)), global_def_action)
    plt.xlabel("Defense strategy id", fontsize=font_size)
    plt.ylabel("Def# of timesd", fontsize=font_size)
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/dsFreq_dsID_A3C.pdf", dpi=figure_dpi)
    plt.show()
    print(global_def_action)

if __name__ == "__main__":
    figure_high = 5
    figure_width = 7.5
    figure_linewidth = 3
    font_size = 20
    figure_dpi = 100

    # print DRL_main result
    # reward_training()
    # mission_condition()
    # defense_stra_distribution()

    # print A3C_try result
    reward_training_A3C()
    t_step_A3C()
    defense_stra_distribution_A3C()