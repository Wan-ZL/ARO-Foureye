'''
Project     ：Drone-DRL-HT 
File        ：new_figure_generate.py
Author      ：Zelin Wan
Date        ：6/2/23
Description : 
'''

import os
import numpy as np

import matplotlib

from matplotlib import pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from scipy import interpolate
from scipy.interpolate import make_interp_spline, interp1d, Akima1DInterpolator
from scipy.signal import savgol_filter

scheme_list = ['scheme_att', 'scheme_def', 'scheme_DefAtt', 'scheme_random']
HEU_usage_list = ['HEU_def=False_att=False', 'HEU_def=True_att=False', 'HEU_def=False_att=True', 'HEU_def=True_att=True']

def_fixed_path = {'F-F': 'scheme_fixed-fixed', 'HT-F': 'scheme_HEU-fixed',
                  'DRL-F': 'scheme_DRL-fixed', 'HDRL-F': 'scheme_HT-DRL-fixed'}
def_HT_path = {'F-HT': 'scheme_fixed-HEU', 'HT-HT': 'scheme_HEU-HEU',
               'DLR-HT': 'scheme_DRL-HEU', 'HDRL-HT': 'scheme_HT-DRL-HEU'}
def_DRL_path = {'F-DRL': 'scheme_fixed-DRL', 'HT-DRL': 'scheme_HEU-DRL',
                'DRL-DRL': 'scheme_DRL-DRL', 'HDRL-DRL': 'scheme_HT-DRL-DRL'}
def_HDRL_path = {'F-HDRL': 'scheme_fixed-HT-DRL', 'HT-HDRL': 'scheme_HEU-HT-DRL',
                 'DRL-HDRL': 'scheme_DRL-HT-DRL', 'HDRL-HDRL': 'scheme_HT-DRL-HT-DRL'}
att_fixed_path = {'F': 'scheme_fixed-fixed', 'HT': 'scheme_fixed-HEU',
                    'DRL': 'scheme_fixed-DRL', 'HT-DRL': 'scheme_fixed-HT-DRL'}
att_HT_path = {'F': 'scheme_HEU-fixed', 'HT': 'scheme_HEU-HEU',
               'DRL': 'scheme_HEU-DRL', 'HT-DRL': 'scheme_HEU-HT-DRL'}
att_DRL_path = {'F': 'scheme_DRL-fixed', 'HT': 'scheme_DRL-HEU',
                'DRL': 'scheme_DRL-DRL', 'HT-DRL': 'scheme_DRL-HT-DRL'}
att_HDRL_path = {'F': 'scheme_HT-DRL-fixed', 'HT': 'scheme_HT-DRL-HEU',
                 'DRL': 'scheme_HT-DRL-DRL', 'HT-DRL': 'scheme_HT-DRL-HT-DRL'}
compare_exist_att_fixed_path = {'HD-F': 'scheme_fixed-fixed', 'IDS': 'scheme_fixed-IDS', 'CD': 'scheme_fixed-CD',
                                'HD-HT-DRL': 'scheme_fixed-HT-DRL', 'No-Defense': 'scheme_fixed-No-Defense'}
compare_exist_att_HT_path = {'HD-F': 'scheme_HEU-fixed', 'IDS': 'scheme_HEU-IDS', 'CD': 'scheme_HEU-CD',
                             'HD-HT-DRL': 'scheme_HEU-HT-DRL', 'No-Defense': 'scheme_HEU-No-Defense'}
compare_exist_att_DRL_path = {'HD-F': 'scheme_DRL-fixed', 'IDS': 'scheme_DRL-IDS', 'CD': 'scheme_DRL-CD',
                              'HD-HT-DRL': 'scheme_DRL-HT-DRL', 'No-Defense': 'scheme_DRL-No-Defense'}
compare_exist_att_HDRL_path = {'HD-F': 'scheme_HT-DRL-fixed', 'IDS': 'scheme_HT-DRL-IDS',
                               'CD': 'scheme_HT-DRL-CD', 'HD-HT-DRL': 'scheme_HT-DRL-HT-DRL',
                               'No-Defense': 'scheme_HT-DRL-No-Defense'}

# code for MobiHoc workshop paper
compare_exist_att_fixed_path_MobiHoc = {'HD-F': 'scheme_fixed-fixed', 'IDS': 'scheme_fixed-IDS', 'CD': 'scheme_fixed-CD',
                                        'HD-DRL': 'scheme_fixed-DRL', 'HD-GT': 'scheme_fixed-HEU', 'No-Defense': 'scheme_fixed-No-Defense'}
compare_exist_att_HT_path_MobiHoc = {'HD-F': 'scheme_HEU-fixed', 'IDS': 'scheme_HEU-IDS', 'CD': 'scheme_HEU-CD',
                                        'HD-DRL': 'scheme_HEU-DRL', 'HD-GT': 'scheme_HEU-HEU', 'No-Defense': 'scheme_HEU-No-Defense'}
compare_exist_att_DRL_path_MobiHoc = {'HD-F': 'scheme_DRL-fixed', 'IDS': 'scheme_DRL-IDS', 'CD': 'scheme_DRL-CD',
                                      'HD-DRL': 'scheme_DRL-DRL', 'HD-GT': 'scheme_DRL-HEU', 'No-Defense': 'scheme_DRL-No-Defense'}



folder_for_mean = 'mean'
tag = 'Accumulated Reward/Defender'
metric_tags = {'Accumulated Reward of Defender': 'Accumulated Reward/Defender',
               'Accumulated Reward of Attacker': 'Accumulated Reward/Attacker',
               'Drones\' Energy Consumption': 'Energy/Energy Consumption',
               # 'Drones\' Energy Consumption (only MDs)': 'Energy/Average Energy MD',
               # 'Drones\' Energy Consumption (only HDs)': 'Energy/Average Energy HD',
               'Ratio of Completed Mission Tasks': 'Ratio of Mission Completion',
               'Number of Active, Connected Drones': 'Drone Number/Average Connect_RLD MD+HD Number'}
metric_to_name = {'Defender Strategy': 'Defender Strategy Frequency',
                  'Attacker Strategy': 'Attacker Strategy Frequency',}
metric_to_latex = {'Accumulated Reward of Defender': '$\mathcal{G}^D$',
                   'Accumulated Reward of Attacker': '$\mathcal{G}^A$',
                   'Ratio of Completed Mission Tasks': '$\mathcal{R}_{MC}$',
                   'Drones\' Energy Consumption': '$\mathcal{EC}$',
                   'Number of Active, Connected Drones': '$\mathcal{N}_{AC}$'}

display_range = {'min': 60, 'max':200} # {'min': 66, 'max':350}
color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
transparent_value = 0.5
smooth_window_size = 30
smooth_polyorder = 2

# ==================== Display Setting for Figure ====================
font_size = 17
figure_high = 4.8 # default: 4.8
figure_width = 6.4 # default: 6.4
figure_linewidth = 3
figure_dpi = 100
legend_size = 18  # 18
axis_size = 15
marker_size = 12
marker_list = ["p", "d", "v", "x", "s", "*", "1", "."]
bar_pattern = ["|", "\\", "/", "+", "-", ".", "*", "x", "o", "O"]
linestyle_list = ["-", "--", "-.", ":", "solid", "dashed", "dashdot", "dotted"]
strategy_number = 8
max_x_length = 60
use_legend = False

def log_reader(path, tag):
    event_acc = EventAccumulator(path)
    event_acc.Reload()
    # get tag data from event_acc
    x_set = []
    y_set = []
    for event in event_acc.Scalars(tag):
        if event.step < display_range['min']:
            continue
        elif event.step > display_range['max']:
            break
        else:
            x_set.append(event.step-display_range['min'])
            y_set.append(event.value)
    return x_set, y_set

def draw_performance_figure(dir_path, def_path_dict, tag, tag_name, title, extra_name=''):
    index = 0
    lines = []
    plt.figure(figsize=(figure_width, figure_high))
    for scheme_label, scheme_path in def_path_dict.items():
        print("scheme_label: ", scheme_label)
        print("path: ", scheme_path)
        x_set, y_set = log_reader(dir_path+'/150_5_5/'+scheme_path+'/'+folder_for_mean, tag)
        # draw figure
        plt.plot(x_set, y_set, color=color_list[index], alpha=transparent_value)
        # draw smooth curve
        # poly = np.polyfit(x_set, y_set, 20)
        # y_smooth = np.poly1d(poly)(x_set)
        y_smooth = savgol_filter(y_set, smooth_window_size, smooth_polyorder)
        line, = plt.plot(x_set, y_smooth, label=scheme_label, linewidth=figure_linewidth, color=color_list[index])
        lines.append(line)
        index += 1

    plt.xlabel('Episode', fontsize=font_size)
    plt.ylabel(metric_to_latex[tag_name], fontsize=font_size*1.5)
    plt.xticks(fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    # plt.title(title, fontsize=font_size)
    # plt.legend(title='Att-Def')
    plt.tight_layout()
    # save figure
    path = 'figures/Performance_Analysis/'
    os.makedirs(path, exist_ok=True)
    plt.savefig(path + title + '-' + tag_name + extra_name + '.png')
    plt.savefig(path + title + '-' + tag_name + extra_name + '.pdf')
    plt.show()

    # # Draw legend
    fig = plt.figure()
    if extra_name == '':
        figlegend = plt.figure(figsize=(6.7, 0.7))
        ncol_size = 4
    else:
        figlegend = plt.figure(figsize=(10.2, 0.7))
        ncol_size = 5
    # figlegend.legend(title='Defender action:',handles=lines, prop={"size": legend_size}, ncol=4)
    figlegend.legend(handles=lines, prop={"size": legend_size}, ncol=ncol_size)

    # fig.show()
    figlegend.show()
    figlegend.savefig(path + 'legend' + tag_name + extra_name + '.png')
    figlegend.savefig(path + 'legend' + tag_name + extra_name + '.pdf')

# code for MobiHoc workshop paper
def draw_performance_figure_MobiHoc(dir_path, def_path_dict, tag, tag_name, title, extra_name=''):
    index = 0
    lines = []
    plt.figure(figsize=(figure_width, figure_high))
    for scheme_label, scheme_path in def_path_dict.items():
        print("scheme_label: ", scheme_label)
        print("path: ", scheme_path)
        x_set, y_set = log_reader(dir_path+'/150_5_5/'+scheme_path+'/'+folder_for_mean, tag)
        # draw figure
        plt.plot(x_set, y_set, color=color_list[index], alpha=transparent_value)
        # draw smooth curve
        # poly = np.polyfit(x_set, y_set, 20)
        # y_smooth = np.poly1d(poly)(x_set)
        y_smooth = savgol_filter(y_set, smooth_window_size, smooth_polyorder)
        line, = plt.plot(x_set, y_smooth, label=scheme_label, linewidth=figure_linewidth, color=color_list[index])
        lines.append(line)
        index += 1

    plt.xlabel('Episode', fontsize=font_size)
    plt.ylabel(metric_to_latex[tag_name], fontsize=font_size*1.5)
    plt.xticks(fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    # plt.title(title, fontsize=font_size)
    # plt.legend(title='Att-Def')
    plt.tight_layout()
    # save figure
    path = 'figures/Performance_Analysis_MobiHoc/'
    os.makedirs(path, exist_ok=True)
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.savefig(path + title + '-' + tag_name + extra_name + '.png')
    plt.savefig(path + title + '-' + tag_name + extra_name + '.pdf')
    plt.show()

    # # Draw legend
    fig = plt.figure()
    if extra_name == '':
        figlegend = plt.figure(figsize=(6.7, 0.7))
        ncol_size = 4
    else:
        figlegend = plt.figure(figsize=(11.83, 0.7))
        ncol_size = 5
    # figlegend.legend(title='Defender action:',handles=lines, prop={"size": legend_size}, ncol=4)
    figlegend.legend(handles=lines, prop={"size": legend_size}, ncol=6)

    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    # fig.show()
    figlegend.show()
    figlegend.savefig(path + 'legend' + tag_name + extra_name + '.png')
    figlegend.savefig(path + 'legend' + tag_name + extra_name + '.pdf')


def draw_sensitivity_analysis_figure(dir_path, def_path_dict, tag, tag_name, sensitivity_dict, latex_name, title):
    episode_to_display = 138 #500
    index = 0
    lines = []
    plt.figure(figsize=(figure_width, figure_high))
    for scheme_label, scheme_path in def_path_dict.items():
        sensitivity_Y_set = []
        for sensitivity in sensitivity_dict.values():
            x_set, y_set = log_reader(dir_path+'/'+sensitivity+'/'+scheme_path+'/'+folder_for_mean, tag)
            y_smooth = savgol_filter(y_set, smooth_window_size, smooth_polyorder)
            y_mean_display = y_smooth[episode_to_display]
            sensitivity_Y_set.append(y_mean_display)
        # draw curve
        line, = plt.plot(sensitivity_dict.keys(), sensitivity_Y_set,  label=scheme_label, linewidth=figure_linewidth,
                         color=color_list[index], marker=marker_list[index], markersize=marker_size,
                         linestyle=linestyle_list[index])
        lines.append(line)
        index += 1

    plt.xlabel(latex_name['latex'], fontsize=font_size)
    plt.ylabel(metric_to_latex[tag_name], fontsize=font_size*1.5)
    plt.xticks(fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    # plt.title(title, fontsize=font_size)
    # plt.legend(title='Att-Def')
    plt.tight_layout()
    # save figure
    path = 'figures/Sensitivity_Analysis/'
    os.makedirs(path, exist_ok=True)
    plt.savefig(path + 'SenseAnalysis-' + latex_name['name'] + '-' + title + '-' + tag_name + '.png')
    plt.savefig(path + 'SenseAnalysis-' + latex_name['name'] + '-' + title + '-' + tag_name + '.pdf')
    plt.show()

    # # Draw legend
    fig = plt.figure()
    figlegend = plt.figure(figsize=(6.7, 0.7))
    # figlegend.legend(title='Defender action:',handles=lines, prop={"size": legend_size}, ncol=4)
    figlegend.legend(handles=lines, prop={"size": legend_size}, ncol=4)

    # fig.show()
    figlegend.show()
    figlegend.savefig(path + 'SenseAnalysis-' + 'legend' + tag_name + '.png')
    figlegend.savefig(path + 'SenseAnalysis-' + 'legend' + tag_name + '.pdf')


def draw_strategy_frequency_bar_figure(dir_path, scheme_path_dict, metric, metric_name, title, episode_to_show=11, extra_name=''):
    print(dir_path)
    print(scheme_path_dict)
    print(metric)
    index = 1
    strategy_num = 10
    lines = []
    plt.figure(figsize=(figure_width, figure_high))
    for scheme_label, scheme_path in scheme_path_dict.items():
        print("scheme_label: ", scheme_label)
        print("path: ", scheme_path)

        # skip the defender's Fixed scheme
        if scheme_label == 'F':
            continue

        bar_y = []
        for strategy_id in range(strategy_num):
            strat_id_tag = metric + '/(' + str(strategy_id) + ') Freq.'
            x_set, y_set = log_reader(dir_path + '/150_5_5/' + scheme_path + '/' + folder_for_mean, strat_id_tag)
            # show the mean of last 20 episodes
            bar_y.append(np.mean(y_set[episode_to_show-10:episode_to_show]))
        # draw bar figure
        print("bar_y: ", bar_y)
        bar_x = np.arange(strategy_num) + 0.6 + index * 0.2
        print(bar_x)
        line = plt.bar(bar_x, bar_y, width=0.2, label=scheme_label, color=color_list[index])

        # draw curve figure that connect the bars
        akima = Akima1DInterpolator(bar_x, bar_y)
        bar_x_new = np.linspace(bar_x.min(), bar_x.max(), 500)
        interpolated_y = akima(bar_x_new)
        # Ensure values don't go below zero
        interpolated_y = np.clip(interpolated_y, 0, None)
        plt.plot(bar_x_new, interpolated_y, label=scheme_label, linewidth=figure_linewidth, color=color_list[index],
                 alpha=transparent_value)

        lines.append(line)
        index += 1

    plt.xlabel(metric, fontsize=font_size)
    plt.ylabel(metric_name, fontsize=font_size)
    # show x aixs from 1 to 10
    plt.xticks(np.arange(1, strategy_num + 1, 1), fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    plt.tight_layout()
    # save figure
    path = 'figures/Performance_Analysis/'
    os.makedirs(path, exist_ok=True)
    if episode_to_show == -1:
        episode_name = 'last'
    else:
        episode_name = str(episode_to_show)
    plt.savefig(path + title + '-' + metric_name + extra_name + '_epi_name_' + episode_name + '.png')
    plt.savefig(path + title + '-' + metric_name + extra_name + '_epi_name_' + episode_name + '.pdf')
    plt.show()

    # # Draw legend
    fig = plt.figure()
    if extra_name == '':
        # figlegend = plt.figure(figsize=(6.7, 0.7))
        # ncol_size = 4
        figlegend = plt.figure(figsize=(5.3, 0.7))
        ncol_size = 3
    else:
        figlegend = plt.figure(figsize=(10.2, 0.7))
        ncol_size = 5
    # figlegend.legend(title='Defender action:',handles=lines, prop={"size": legend_size}, ncol=4)
    figlegend.legend(handles=lines, prop={"size": legend_size}, ncol=ncol_size)

    # fig.show()
    figlegend.show()
    figlegend.savefig(path + 'legend' + metric_to_name[metric] + extra_name + '.png')
    figlegend.savefig(path + 'legend' + metric_to_name[metric] + extra_name + '.pdf')





if __name__ == '__main__':
    dir_path = 'data/tb_reduce_MobiHoc' #'data/tb_reduce_TDSC' #'data/tb_reduce_MobiHoc'
    # draw performance figures
    # for metric_name, metric in metric_tags.items():
    #     # compare proposed method
    #     draw_performance_figure(dir_path, att_fixed_path, metric, metric_name, 'Attacker Fixed')
    #     draw_performance_figure(dir_path, att_HT_path, metric, metric_name, 'Attacker HT')
    #     draw_performance_figure(dir_path, att_DRL_path, metric, metric_name, 'Attacker DRL')
    #     draw_performance_figure(dir_path, att_HDRL_path, metric, metric_name, 'Attacker HT-DRL')
    #     # compare existing method
    #     draw_performance_figure(dir_path, compare_exist_att_fixed_path, metric, metric_name, 'Attacker Fixed', extra_name='-CompareExist')
    #     draw_performance_figure(dir_path, compare_exist_att_HT_path, metric, metric_name, 'Attacker HT', extra_name='-CompareExist')
    #     draw_performance_figure(dir_path, compare_exist_att_DRL_path, metric, metric_name, 'Attacker DRL', extra_name='-CompareExist')
    #     draw_performance_figure(dir_path, compare_exist_att_HDRL_path, metric, metric_name, 'Attacker HT-DRL', extra_name='-CompareExist')

    # draw 'strategy frequency' bar figure
    # for metric, metric_name in metric_to_name.items():
    #     draw_strategy_frequency_bar_figure(dir_path, att_fixed_path, metric, metric_name, 'Attacker Fixed')
    #     draw_strategy_frequency_bar_figure(dir_path, att_HT_path, metric, metric_name, 'Attacker HT')
    #     draw_strategy_frequency_bar_figure(dir_path, att_DRL_path, metric, metric_name, 'Attacker DRL')
    #     draw_strategy_frequency_bar_figure(dir_path, att_HDRL_path, metric, metric_name, 'Attacker HT-DRL')



    # Code for MobiHoc workshop paper
    for metric_name, metric in metric_tags.items():
        draw_performance_figure_MobiHoc(dir_path, compare_exist_att_fixed_path_MobiHoc, metric, metric_name, 'Attacker Fixed', extra_name='-CompareExist')
        draw_performance_figure_MobiHoc(dir_path, compare_exist_att_HT_path_MobiHoc, metric, metric_name, 'Attacker HT', extra_name='-CompareExist')
        draw_performance_figure_MobiHoc(dir_path, compare_exist_att_DRL_path_MobiHoc, metric, metric_name, 'Attacker DRL', extra_name='-CompareExist')


    # draw 'mission time' sensitivity analysis figures
    zeta_sensitivity_set = {'100': '110_5_5', '130': '130_5_5', '150': '150_5_5', '170': '170_5_5', '190': '190_5_5'}
    mission_time_latex = {'name': 'Mission Time', 'latex':  '$T_{M}^{\mathrm{max}}$'}
    # for metric_name, metric in metric_tags.items():
    #     draw_sensitivity_analysis_figure(dir_path, att_fixed_path, metric, metric_name, zeta_sensitivity_set, mission_time_latex, 'Attacker Fixed')
    #     draw_sensitivity_analysis_figure(dir_path, att_HT_path, metric, metric_name, zeta_sensitivity_set, mission_time_latex, 'Attacker HT')
    #     draw_sensitivity_analysis_figure(dir_path, att_DRL_path, metric, metric_name, zeta_sensitivity_set, mission_time_latex, 'Attacker DRL')
    #     draw_sensitivity_analysis_figure(dir_path, att_HDRL_path, metric, metric_name, zeta_sensitivity_set, mission_time_latex, 'Attacker HT-DRL')
    #
    # # draw 'attack budget' sensitivity analysis figures
    attack_budget_sensitivity_set = {'1': '150_1_5', '3': '150_3_5', '5': '150_5_5', '7': '150_7_5', '9': '150_9_5'}
    attack_budget_latex = {'name': 'Attack Budget', 'latex': '$\zeta$'}
    # for metric_name, metric in metric_tags.items():
    #     draw_sensitivity_analysis_figure(dir_path, att_fixed_path, metric, metric_name, attack_budget_sensitivity_set, attack_budget_latex, 'Attacker Fixed')
    #     draw_sensitivity_analysis_figure(dir_path, att_HT_path, metric, metric_name, attack_budget_sensitivity_set, attack_budget_latex, 'Attacker HT')
    #     draw_sensitivity_analysis_figure(dir_path, att_DRL_path, metric, metric_name, attack_budget_sensitivity_set, attack_budget_latex, 'Attacker DRL')
    #     draw_sensitivity_analysis_figure(dir_path, att_HDRL_path, metric, metric_name, attack_budget_sensitivity_set, attack_budget_latex, 'Attacker HT-DRL')




