from Foureye_parallel_simulation import *

display = False

if __name__ == '__main__':
    start = time.time()
    simulation_time = 100
    # game_start()
    run_sumulation_fixed_setting("DD-IPI", True, True, simulation_time)
    # run_sumulation_fixed_setting("DD-PI", True, False, simulation_time)
    # run_sumulation_fixed_setting("No-DD-IPI", False, True, simulation_time)
    # run_sumulation_fixed_setting("No-DD-PI", False, False, simulation_time)

    # Varying Vulnerability Upper Bound
    # run_sumulation_group_varying_vul("DD-IPI", True, True, simulation_time)
    # run_sumulation_group_varying_vul("DD-PI", True, False, simulation_time)
    # run_sumulation_group_varying_vul("No-DD-IPI", False, True, simulation_time)
    # run_sumulation_group_varying_vul("No-DD-PI", False, False, simulation_time)

    # Varying Th_risk
    # run_sumulation_group_varying_universal("DD-IPI", True, True, simulation_time, "Th_risk", [0.1, 0.2, 0.3, 0.4, 0.5])
    # run_sumulation_group_varying_universal("DD-PI", True, False, simulation_time, "Th_risk", [0.1, 0.2, 0.3, 0.4, 0.5])
    # run_sumulation_group_varying_universal("No-DD-IPI", False, True, simulation_time, "Th_risk", [0.1, 0.2, 0.3, 0.4, 0.5])
    # run_sumulation_group_varying_universal("No-DD-PI", False, False, simulation_time, "Th_risk", [0.1, 0.2, 0.3, 0.4, 0.5])

    # varying _lambda
    # run_sumulation_group_varying_universal("DD-IPI", True, True, simulation_time, "_lambda", [0.6, 0.7, 0.8, 0.9, 1])
    # run_sumulation_group_varying_universal("DD-PI", True, False, simulation_time, "_lambda", [0.6, 0.7, 0.8, 0.9, 1])
    # run_sumulation_group_varying_universal("No-DD-IPI", False, True, simulation_time, "_lambda", [0.6, 0.7, 0.8, 0.9, 1])
    # run_sumulation_group_varying_universal("No-DD-PI", False, False, simulation_time, "_lambda", [0.6, 0.7, 0.8, 0.9, 1])

    # varying _mu
    # run_sumulation_group_varying_universal("DD-IPI", True, True, simulation_time, "mu", [6, 7, 8, 9, 10])
    # run_sumulation_group_varying_universal("DD-PI", True, False, simulation_time, "mu", [6, 7, 8, 9, 10])
    # run_sumulation_group_varying_universal("No-DD-IPI", False, True, simulation_time, "mu", [6, 7, 8, 9, 10])
    # run_sumulation_group_varying_universal("No-DD-PI", False, False, simulation_time, "mu", [6, 7, 8, 9, 10])

    # varying Rho_1
    # run_sumulation_group_varying_universal("DD-IPI", True, True, simulation_time, "SF_thres_1", [0.1, 0.2, 0.33, 0.4, 0.5])
    # run_sumulation_group_varying_universal("DD-PI", True, False, simulation_time, "SF_thres_1", [0.1, 0.2, 0.33, 0.4, 0.5])
    # run_sumulation_group_varying_universal("No-DD-IPI", False, True, simulation_time, "SF_thres_1", [0.1, 0.2, 0.33, 0.4, 0.5])
    # run_sumulation_group_varying_universal("No-DD-PI", False, False, simulation_time, "SF_thres_1", [0.1, 0.2, 0.33, 0.4, 0.5])

    # varying Rho_2
    # run_sumulation_group_varying_universal("DD-IPI", True, True, simulation_time, "SF_thres_2", [0.3, 0.4, 0.5, 0.6, 0.7])
    # run_sumulation_group_varying_universal("DD-PI", True, False, simulation_time, "SF_thres_2", [0.3, 0.4, 0.5, 0.6, 0.7])
    # run_sumulation_group_varying_universal("No-DD-IPI", False, True, simulation_time, "SF_thres_2", [0.3, 0.4, 0.5, 0.6, 0.7])
    # run_sumulation_group_varying_universal("No-DD-PI", False, False, simulation_time, "SF_thres_2", [0.3, 0.4, 0.5, 0.6, 0.7])

    # varying attacker detectability upper bound
    # run_sumulation_group_varying_universal("DD-IPI", True, True, simulation_time, "att_detect_UpBod", [0.3, 0.4, 0.5, 0.6, 0.7])
    # run_sumulation_group_varying_universal("DD-PI", True, False, simulation_time, "att_detect_UpBod", [0.3, 0.4, 0.5, 0.6, 0.7])
    # run_sumulation_group_varying_universal("No-DD-IPI", False, True, simulation_time, "att_detect_UpBod", [0.3, 0.4, 0.5, 0.6, 0.7])
    # run_sumulation_group_varying_universal("No-DD-PI", False, False, simulation_time, "att_detect_UpBod", [0.3, 0.4, 0.5, 0.6, 0.7])

    print("Project took", time.time() - start, "seconds.")
    # time.sleep(10)
    # try:
    #     os.system('say "your program has finished"')
    #     os._exit(0)
    # except:
    #     print("Your command not found: say")
