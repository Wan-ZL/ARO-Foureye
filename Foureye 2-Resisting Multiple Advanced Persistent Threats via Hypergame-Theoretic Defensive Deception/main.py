# Python Version: 3.6.12
# link all files
from game_function import *
import concurrent.futures
import os
import time
import multiprocessing
import graph_function
from datetime import datetime

display = False


def game_start(simulation_id=1,
               DD_using=True,
               uncertain_scheme_att=True,
               uncertain_scheme_def=True,
               decision_scheme=1,
               scheme_name = "DD-IPI",
               web_data_upper_vul=7,
               Iot_upper_vul=5,
               att_arr_prob = 0.1,
               vary_name=None,
               vary_value=0):
    print(
        f"Start Simulation {simulation_id}, DD_using={DD_using}, uncertain_scheme={[uncertain_scheme_att, uncertain_scheme_def]}"
    )
    # np.seterr(divide='ignore', invalid='ignore')  # for remove divide zero warning

    game_continue = True

    game = game_class(simulation_id, DD_using, uncertain_scheme_att, uncertain_scheme_def, decision_scheme,scheme_name,
                      web_data_upper_vul, Iot_upper_vul, att_arr_prob, vary_name, vary_value)
    # show_all_nodes(game.graph.network)
    # return
    while (not game.game_over):
        print(game.lifetime)
        game.defender_round()
        game.evict_reason_history[1] += game.count_number_of_evicted_attacker()
        game.attacker_round(simulation_id)
        game.evict_reason_history[2] += game.count_number_of_evicted_attacker()
        game.experiment_saving()

        game.NIDS_detect()
        game.IDS_IRS_evict()
        game.evict_reason_history[3] += game.count_number_of_evicted_attacker()
        game.IRS_recover()

        reason_box = [None]
        if is_system_fail(game.graph, reason_box):
            print(f"Sim {simulation_id} SYSTEM FAIL \U0001F480")
            print(f"Sim {simulation_id} GAME OVER")
            game.game_over = True
            game.SysFail[reason_box[0]] = True

        game.prepare_for_next_game()

        game.new_attacker(simulation_id, game.defender)

        # Test: vulnerability of compromised node right before game over
        if game.game_over:
            vulnerability_set_of_compromised_node = []
            for node_id in game.graph.network.nodes():
                if game.graph.network.nodes[node_id]["compromised_status"]:
                    vulnerability_set_of_compromised_node.append(
                        game.graph.network.nodes[node_id]["vulnerability"])
            print(f"Sorted vulnerability of Compromised Nodes: {sorted(vulnerability_set_of_compromised_node)}")


        # print(np.mean(list(nx.get_node_attributes(game.graph.network, "normalized_vulnerability").values())))
    return game


def run_sumulation_group_1(current_scheme, DD_using, uncertain_scheme_att, uncertain_scheme_def, decision_scheme, simulation_time):
    MTTSF_all_result = 0
    def_uncertainty_all_result = {}
    att_uncertainty_all_result = {}
    Time_to_SF_all_result = {}
    att_HEU_all_result = {}
    def_HEU_all_result = {}
    AHEU_per_Strategy_all_result = {}
    DHEU_per_Strategy_all_result = {}
    att_strategy_count_result = {}
    def_strategy_count_result = {}
    FPR_all_result = {}
    TPR_all_result = {}
    TP_all_result = {}
    FN_all_result = {}
    TN_all_result = {}
    FP_all_result = {}
    att_cost_all_result = {}
    def_cost_all_result = {}
    criticality_all_result = {}
    evict_reason_all_result = {}
    SysFail_reason = [0] * 3
    att_EU_C_all_result = {}
    att_EU_CMS_all_result = {}
    def_EU_C_all_result = {}
    def_EU_CMS_all_result = {}
    att_impact_all_result = {}
    def_impact_all_result = {}
    att_HEU_DD_IPI_all_result = {}
    def_HEU_DD_IPI_all_result = {}
    NIDS_eviction_all_result = {}
    number_of_attacker_all_result = {}
    att_CKC_all_result = {}
    compromise_probability_all_result = {}
    number_of_inside_attacker_all_result = {}
    all_result_after_each_game_all_result = {}
    hitting_probability_all_result = {}


    results = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for i in range(simulation_time):
            future = executor.submit(game_start, i, DD_using,
                                     uncertain_scheme_att, uncertain_scheme_def, decision_scheme, current_scheme)  # scheme change here
            results.append(future)

        index = 0

        for future in results:
            # MTTSF
            MTTSF_all_result += future.result().lifetime
            # New Attacker
            Time_to_SF_all_result[index] = future.result().lifetime
            # HEU
            att_HEU_all_result[index] = future.result().att_HEU_history
            def_HEU_all_result[index] = future.result().def_HEU_history
            # AHEU/DHEU per Strategy
            AHEU_per_Strategy_all_result[index] = future.result().AHEU_per_Strategy_History
            DHEU_per_Strategy_all_result[index] = future.result().DHEU_per_Strategy_History
            # Strategy Counter
            att_strategy_count_result[index] = future.result(
            ).att_strategy_counter
            def_strategy_count_result[index] = future.result(
            ).def_strategy_counter
            # Uncertainty
            def_uncertainty_all_result[index] = future.result(
            ).def_uncertainty_history
            att_uncertainty_all_result[index] = future.result(
            ).att_uncertainty_history
            # TPR & FPR
            FPR_all_result[index] = future.result().FPR_history
            TPR_all_result[index] = future.result().TPR_history
            # Cost
            att_cost_all_result[index] = future.result().att_cost_history
            def_cost_all_result[index] = future.result().def_cost_history
            # Criticality
            criticality_all_result[index] = future.result().criticality_hisotry
            # Evict attacker reason
            evict_reason_all_result[index] = future.result(
            ).evict_reason_history
            # System Fail reason
            if future.result().SysFail[0]:
                SysFail_reason[0] += 1  # [att_strat, system_fail]
            elif future.result().SysFail[1]:
                SysFail_reason[1] += 1
            elif future.result().SysFail[2]:
                SysFail_reason[2] += 1
            # EU_C & EU_CMS
            att_EU_C_all_result[index] = np.delete(future.result().att_EU_C, 0,
                                                   0)
            att_EU_CMS_all_result[index] = np.delete(
                future.result().att_EU_CMS, 0, 0)
            def_EU_C_all_result[index] = np.delete(future.result().def_EU_C, 0,
                                                   0)
            def_EU_CMS_all_result[index] = np.delete(
                future.result().def_EU_CMS, 0, 0)
            # attacker & defender impact
            att_impact_all_result[index] = np.delete(
                future.result().att_impact, 0, 0)
            def_impact_all_result[index] = np.delete(
                future.result().def_impact, 0, 0)
            # HEU in DD IPI
            att_HEU_DD_IPI_all_result[index] = np.delete(
                future.result().att_HEU_DD_IPI, 0, 0)
            def_HEU_DD_IPI_all_result[index] = np.delete(
                future.result().def_HEU_DD_IPI, 0, 0)
            # NIDS evict Bad or Good
            NIDS_eviction_all_result[index] = future.result().NIDS_eviction
            # Number of attacker
            number_of_attacker_all_result[index] = future.result().att_number
            # attacker CKC
            att_CKC_all_result[index] = future.result().att_CKC
            # compromised probability for AS2
            compromise_probability_all_result[index] = future.result().compromise_probability
            # inside attacker counter
            number_of_inside_attacker_all_result[index] = future.result().number_of_inside_attacker
            # data for ML model training
            all_result_after_each_game_all_result[index] = future.result().all_result_after_each_game
            # hitting probability for Hypergame Nash Equilibrium
            hitting_probability_all_result[index] = future.result().hitting_result


            index += 1

    # SAVE to FILE
    os.makedirs("data/" + current_scheme + "/R0", exist_ok=True)
    the_file = open("data/" + current_scheme + "/R0/Time_to_SF.pkl", "wb+")
    pickle.dump(Time_to_SF_all_result, the_file)
    the_file.close()

    # HEU
    os.makedirs("data/" + current_scheme + "/R1", exist_ok=True)
    the_file = open("data/" + current_scheme + "/R1/att_HEU.pkl", "wb+")
    pickle.dump(att_HEU_all_result, the_file)
    the_file.close()
    the_file = open("data/" + current_scheme + "/R1/def_HEU.pkl", "wb+")
    pickle.dump(def_HEU_all_result, the_file)
    the_file.close()

    # Strategy Counter
    os.makedirs("data/" + current_scheme + "/R2", exist_ok=True)
    the_file = open("data/" + current_scheme + "/R2/att_strategy_counter.pkl",
                    "wb+")
    pickle.dump(att_strategy_count_result, the_file)
    the_file.close()
    the_file = open("data/" + current_scheme + "/R2/def_strategy_counter.pkl",
                    "wb+")
    pickle.dump(def_strategy_count_result, the_file)
    the_file.close()

    # uncertainty
    os.makedirs("data/" + current_scheme + "/R3", exist_ok=True)
    the_file = open("data/" + current_scheme + "/R3/defender_uncertainty.pkl",
                    "wb+")
    pickle.dump(def_uncertainty_all_result, the_file)
    the_file.close()
    the_file = open("data/" + current_scheme + "/R3/attacker_uncertainty.pkl",
                    "wb+")
    pickle.dump(att_uncertainty_all_result, the_file)
    the_file.close()

    # TPR & FPR
    os.makedirs("data/" + current_scheme + "/R4", exist_ok=True)
    the_file = open("data/" + current_scheme + "/R4/FPR.pkl", "wb+")
    pickle.dump(FPR_all_result, the_file)
    the_file.close()
    the_file = open("data/" + current_scheme + "/R4/TPR.pkl", "wb+")
    pickle.dump(TPR_all_result, the_file)
    the_file.close()

    # MTTSF
    os.makedirs("data/" + current_scheme + "/R5", exist_ok=True)
    the_file = open("data/" + current_scheme + "/R5/MTTSF.pkl", "wb+")
    pickle.dump(MTTSF_all_result, the_file)
    the_file.close()

    # Cost
    os.makedirs("data/" + current_scheme + "/R6", exist_ok=True)
    the_file = open("data/" + current_scheme + "/R6/att_cost.pkl", "wb+")
    pickle.dump(att_cost_all_result, the_file)
    the_file.close()
    the_file = open("data/" + current_scheme + "/R6/def_cost.pkl", "wb+")
    pickle.dump(def_cost_all_result, the_file)
    the_file.close()

    # Uncertainty
    os.makedirs("data/" + current_scheme + "/R7", exist_ok=True)
    the_file = open("data/" + current_scheme + "/R7/att_uncertainty.pkl",
                    "wb+")
    pickle.dump(att_uncertainty_all_result, the_file)
    the_file.close()
    the_file = open("data/" + current_scheme + "/R7/def_uncertainty.pkl",
                    "wb+")
    pickle.dump(def_uncertainty_all_result, the_file)
    the_file.close()

    # attacker number
    os.makedirs("data/" + current_scheme + "/R8", exist_ok=True)
    the_file = open("data/" + current_scheme + "/R8/number_of_att.pkl",
                    "wb+")
    pickle.dump(number_of_attacker_all_result, the_file)
    the_file.close()

    # attacker CKC
    os.makedirs("data/" + current_scheme + "/R9", exist_ok=True)
    the_file = open("data/" + current_scheme + "/R9/att_CKC.pkl",
                    "wb+")
    pickle.dump(att_CKC_all_result, the_file)
    the_file.close()

    # ////////////////////# ////////////////////# ////////////////////

    # Criticality
    os.makedirs("data/" + current_scheme + "/R_self_1", exist_ok=True)
    the_file = open("data/" + current_scheme + "/R_self_1/criticality.pkl",
                    "wb+")
    pickle.dump(criticality_all_result, the_file)
    the_file.close()

    # Evict attacker reason
    os.makedirs("data/" + current_scheme + "/R_self_2", exist_ok=True)
    the_file = open("data/" + current_scheme + "/R_self_2/evict_reason.pkl",
                    "wb+")
    pickle.dump(evict_reason_all_result, the_file)
    the_file.close()

    # System Failure reason
    os.makedirs("data/" + current_scheme + "/R_self_3", exist_ok=True)
    the_file = open("data/" + current_scheme + "/R_self_3/system_fail.pkl",
                    "wb+")
    pickle.dump(SysFail_reason, the_file)
    the_file.close()

    # EU_C & EU_CMS
    os.makedirs("data/" + current_scheme + "/R_self_4", exist_ok=True)
    the_file = open("data/" + current_scheme + "/R_self_4/att_EU_C.pkl", "wb+")
    pickle.dump(att_EU_C_all_result, the_file)
    the_file.close()
    the_file = open("data/" + current_scheme + "/R_self_4/att_EU_CMS.pkl",
                    "wb+")
    pickle.dump(att_EU_CMS_all_result, the_file)
    the_file.close()
    the_file = open("data/" + current_scheme + "/R_self_4/def_EU_C.pkl", "wb+")
    pickle.dump(def_EU_C_all_result, the_file)
    the_file.close()
    the_file = open("data/" + current_scheme + "/R_self_4/def_EU_CMS.pkl",
                    "wb+")
    pickle.dump(def_EU_CMS_all_result, the_file)
    the_file.close()

    # attacker & defender impact
    the_file = open("data/" + current_scheme + "/R_self_4/att_impact.pkl",
                    "wb+")
    pickle.dump(att_impact_all_result, the_file)
    the_file.close()
    the_file = open("data/" + current_scheme + "/R_self_4/def_impact.pkl",
                    "wb+")
    pickle.dump(def_impact_all_result, the_file)
    the_file.close()

    # HEU in DD IPI
    the_file = open("data/" + current_scheme + "/R_self_4/att_HEU_DD_IPI.pkl",
                    "wb+")
    pickle.dump(att_HEU_DD_IPI_all_result, the_file)
    the_file.close()
    the_file = open("data/" + current_scheme + "/R_self_4/def_HEU_DD_IPI.pkl",
                    "wb+")
    pickle.dump(def_HEU_DD_IPI_all_result, the_file)
    the_file.close()

    # NIDS evict good or bad
    the_file = open("data/" + current_scheme + "/R_self_4/NIDS_eviction.pkl",
                    "wb+")
    pickle.dump(NIDS_eviction_all_result, the_file)
    the_file.close()

    # AHEU/DHEU per Strategy (HEU of all strategy)
    os.makedirs("data/" + current_scheme + "/R_self_5", exist_ok=True)
    the_file = open("data/" + current_scheme + "/R_self_5/AHEU_for_all_strategy_DD_IPI.pkl", "wb+")
    pickle.dump(AHEU_per_Strategy_all_result, the_file)
    the_file.close()
    the_file = open("data/" + current_scheme + "/R_self_5/DHEU_for_all_strategy_DD_IPI.pkl", "wb+")
    pickle.dump(DHEU_per_Strategy_all_result, the_file)
    the_file.close()

    # compromised probability
    the_file = open("data/" + current_scheme + "/R_self_5/compromise_probability_all_result.pkl", "wb+")
    pickle.dump(compromise_probability_all_result, the_file)
    the_file.close()

    # number of inside attacker
    the_file = open("data/" + current_scheme + "/R_self_5/number_of_inside_attacker_all_result.pkl", "wb+")
    pickle.dump(number_of_inside_attacker_all_result, the_file)
    the_file.close()

    # training and target data for ML model
    os.makedirs("data/trainning_data/" + current_scheme, exist_ok=True)
    # file_list = os.listdir("data/trainning_data/" + current_scheme)
    # file_ID = str(len(file_list) + 1)
    # the_file = open("data/" + current_scheme + "/ML_data/training/all_result_after_each_game_"+file_ID+".pkl", "wb+")
    now = datetime.now()
    dt_string = now.strftime("%H-%M_%d-%m-%Y")
    the_file = open("data/trainning_data/" + current_scheme + "/all_result_after_each_game_" + dt_string + ".pkl",
                    "wb+")
    pickle.dump(all_result_after_each_game_all_result, the_file)
    the_file.close()

    # Hitting Probability
    the_file = open("data/" + current_scheme + "/R_self_5/hitting_probability.pkl", "wb+")
    pickle.dump(hitting_probability_all_result, the_file)
    the_file.close()


def run_sumulation_group_vary_VUB(current_scheme, DD_using, uncertain_scheme_att, uncertain_scheme_def, decision_scheme, simulation_time):
    vul_range = {}
    MTTSF_all_result = {}
    att_cost_all_result = {}
    def_cost_all_result = {}
    att_HEU_all_result = {}
    def_HEU_all_result = {}
    att_uncertainty_all_result = {}
    def_uncertainty_all_result = {}
    FPR_all_result = {}
    TPR_all_result = {}
    TP_all_result = {}
    FN_all_result = {}
    TN_all_result = {}
    FP_all_result = {}
    att_strategy_all_result = {}
    def_strategy_all_result = {}


    # web_data_SoftVul_range = range(3,7+1)
    # IoT_SoftVul_range = range(1,5+1)
    web_data_SoftVul_range = np.array(range(1, 5 + 1)) * 2
    IoT_SoftVul_range = np.array(range(1, 5 + 1)) * 2
    vul_range[0] = web_data_SoftVul_range
    vul_range[1] = IoT_SoftVul_range

    vul_upper_bound = [2, 4, 6, 8, 10]
    vary_name = "VUB"

    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for vul_index in range(5):
            particular_vul_result = []
            for i in range(simulation_time):
                future = executor.submit(
                    game_start, i, DD_using, uncertain_scheme_att, uncertain_scheme_def, decision_scheme, current_scheme,
                    web_data_upper_vul=web_data_SoftVul_range[vul_index],
                    Iot_upper_vul=IoT_SoftVul_range[vul_index], vary_name=vary_name, vary_value=vul_upper_bound[vul_index])  # scheme change here
                particular_vul_result.append(future)
            results.append(particular_vul_result)

        vul_index = 0
        for particular_vul_result in results:
            MTTSF_temp = {}
            att_cost_temp = {}
            def_cost_temp = {}
            att_HEU_temp = {}
            def_HEU_temp = {}
            att_uncertainty_temp = {}
            def_uncertainty_temp = {}
            FPR_temp = {}
            TPR_temp = {}
            att_stra_temp = {}
            def_stra_temp = {}
            sim_index = 0
            for future in particular_vul_result:
                # change web server and database vul
                # MTTSF
                MTTSF_temp[sim_index] = future.result().lifetime
                # Cost
                att_cost_temp[sim_index] = future.result().att_cost_history
                def_cost_temp[sim_index] = future.result().def_cost_history
                # # HEU
                att_HEU_temp[sim_index] = future.result().att_HEU_history
                def_HEU_temp[sim_index] = future.result().def_HEU_history
                # # Uncertainty
                att_uncertainty_temp[sim_index] = future.result().att_uncertainty_history
                def_uncertainty_temp[sim_index] = future.result().def_uncertainty_history
                # # FPR & TPR
                FPR_temp[sim_index] = future.result().FPR_history
                TPR_temp[sim_index] = future.result().TPR_history
                # # Strategy Usability
                att_stra_temp[sim_index] = future.result().att_strategy_counter
                def_stra_temp[sim_index] = future.result().def_strategy_counter
                sim_index += 1

            MTTSF_all_result[vul_index] = MTTSF_temp
            att_cost_all_result[vul_index] = att_cost_temp
            def_cost_all_result[vul_index] = def_cost_temp
            att_HEU_all_result[vul_index] = att_HEU_temp
            def_HEU_all_result [vul_index]= def_HEU_temp
            att_uncertainty_all_result[vul_index] = att_uncertainty_temp
            def_uncertainty_all_result[vul_index] = def_uncertainty_temp
            FPR_all_result[vul_index] = FPR_temp
            TPR_all_result[vul_index] = TPR_temp
            att_strategy_all_result[vul_index] = att_stra_temp
            def_strategy_all_result[vul_index] = def_stra_temp
            vul_index += 1


    # SAVE to FILE (need to create directory manually)
    os.makedirs("data/" + current_scheme + "/varying_VUB", exist_ok=True)
    # Vul range
    the_file = open("data/" + current_scheme + "/varying_VUB/Vul_Range.pkl", "wb+")
    pickle.dump(vul_range, the_file)
    the_file.close()
    # MTTSF
    the_file = open("data/" + current_scheme + "/varying_VUB/MTTSF.pkl", "wb+")
    pickle.dump(MTTSF_all_result, the_file)
    the_file.close()

    # Cost
    the_file = open("data/" + current_scheme + "/varying_VUB/att_cost.pkl", "wb+")
    pickle.dump(att_cost_all_result, the_file)
    the_file.close()
    the_file = open("data/" + current_scheme + "/varying_VUB/def_cost.pkl", "wb+")
    pickle.dump(def_cost_all_result, the_file)
    the_file.close()

    # HEU
    the_file = open("data/" + current_scheme + "/varying_VUB/att_HEU.pkl", "wb+")
    pickle.dump(att_HEU_all_result, the_file)
    the_file.close()
    the_file = open("data/" + current_scheme + "/varying_VUB/def_HEU.pkl", "wb+")
    pickle.dump(def_HEU_all_result, the_file)
    the_file.close()

    # Uncertainty
    the_file = open("data/" + current_scheme + "/varying_VUB/att_uncertainty.pkl",
                    "wb+")
    pickle.dump(att_uncertainty_all_result, the_file)
    the_file.close()
    the_file = open("data/" + current_scheme + "/varying_VUB/def_uncertainty.pkl",
                    "wb+")
    pickle.dump(def_uncertainty_all_result, the_file)
    the_file.close()

    # FPR & TPR
    the_file = open("data/" + current_scheme + "/varying_VUB/FPR.pkl", "wb+")
    pickle.dump(FPR_all_result, the_file)
    the_file.close()
    the_file = open("data/" + current_scheme + "/varying_VUB/TPR.pkl", "wb+")
    pickle.dump(TPR_all_result, the_file)
    the_file.close()

def run_sumulation_group_vary_AAP(current_scheme, DD_using, uncertain_scheme_att, uncertain_scheme_def, decision_scheme, simulation_time):
    vul_range = {}
    MTTSF_all_result = {}
    att_cost_all_result = {}
    def_cost_all_result = {}
    att_HEU_all_result = {}
    def_HEU_all_result = {}
    att_uncertainty_all_result = {}
    def_uncertainty_all_result = {}
    FPR_all_result = {}
    TPR_all_result = {}
    TP_all_result = {}
    FN_all_result = {}
    TN_all_result = {}
    FP_all_result = {}
    att_strategy_all_result = {}
    def_strategy_all_result = {}

    att_arr_prob = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    vary_name = "AAP"

    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for vul_index in range(len(att_arr_prob)):
            particular_vul_result = []
            for i in range(simulation_time):
                future = executor.submit(
                    game_start, i, DD_using, uncertain_scheme_att, uncertain_scheme_def, decision_scheme, current_scheme,
                    att_arr_prob = att_arr_prob[vul_index], vary_name=vary_name, vary_value=att_arr_prob[vul_index])  # scheme change here
                particular_vul_result.append(future)
            results.append(particular_vul_result)

        vul_index = 0
        for particular_vul_result in results:
            MTTSF_temp = {}
            att_cost_temp = {}
            def_cost_temp = {}
            att_HEU_temp = {}
            def_HEU_temp = {}
            att_uncertainty_temp = {}
            def_uncertainty_temp = {}
            FPR_temp = {}
            TPR_temp = {}
            TP_temp = {}
            FN_temp = {}
            TN_temp = {}
            FP_temp = {}
            att_stra_temp = {}
            def_stra_temp = {}
            sim_index = 0
            for future in particular_vul_result:
                # change web server and database vul
                # MTTSF
                MTTSF_temp[sim_index] = future.result().lifetime
                # Cost
                att_cost_temp[sim_index] = future.result().att_cost_history
                def_cost_temp[sim_index] = future.result().def_cost_history
                # # HEU
                att_HEU_temp[sim_index] = future.result().att_HEU_history
                def_HEU_temp[sim_index] = future.result().def_HEU_history
                # # Uncertainty
                att_uncertainty_temp[sim_index] = future.result().att_uncertainty_history
                def_uncertainty_temp[sim_index] = future.result().def_uncertainty_history
                # # FPR & TPR
                FPR_temp[sim_index] = future.result().FPR_history
                TPR_temp[sim_index] = future.result().TPR_history
                TP_temp[sim_index] = future.result().TP_history
                FN_temp[sim_index] = future.result().FN_history
                TN_temp[sim_index] = future.result().TN_history
                FP_temp [sim_index] = future.result().FP_history
                # # Strategy Usability
                att_stra_temp[sim_index] = future.result().att_strategy_counter
                def_stra_temp[sim_index] = future.result().def_strategy_counter
                sim_index += 1

            MTTSF_all_result[vul_index] = MTTSF_temp
            att_cost_all_result[vul_index] = att_cost_temp
            def_cost_all_result[vul_index] = def_cost_temp
            att_HEU_all_result[vul_index] = att_HEU_temp
            def_HEU_all_result [vul_index]= def_HEU_temp
            att_uncertainty_all_result[vul_index] = att_uncertainty_temp
            def_uncertainty_all_result[vul_index] = def_uncertainty_temp
            FPR_all_result[vul_index] = FPR_temp
            TPR_all_result[vul_index] = TPR_temp
            TP_all_result[vul_index] = TP_temp
            FN_all_result[vul_index] = FN_temp
            TN_all_result[vul_index] = TN_temp
            FP_all_result[vul_index] = FP_temp
            att_strategy_all_result[vul_index] = att_stra_temp
            def_strategy_all_result[vul_index] = def_stra_temp
            vul_index += 1


    # SAVE to FILE (need to create directory manually)
    os.makedirs("data/" + current_scheme + "/varying_AAP", exist_ok=True)
    # Attacker Arrival Probability (AAP)
    the_file = open("data/" + current_scheme + "/varying_AAP/AAP_Range.pkl", "wb+")
    pickle.dump(att_arr_prob, the_file)
    the_file.close()
    # MTTSF
    the_file = open("data/" + current_scheme + "/varying_AAP/MTTSF.pkl", "wb+")
    pickle.dump(MTTSF_all_result, the_file)
    the_file.close()

    # Cost
    the_file = open("data/" + current_scheme + "/varying_AAP/att_cost.pkl", "wb+")
    pickle.dump(att_cost_all_result, the_file)
    the_file.close()
    the_file = open("data/" + current_scheme + "/varying_AAP/def_cost.pkl", "wb+")
    pickle.dump(def_cost_all_result, the_file)
    the_file.close()

    # HEU
    the_file = open("data/" + current_scheme + "/varying_AAP/att_HEU.pkl", "wb+")
    pickle.dump(att_HEU_all_result, the_file)
    the_file.close()
    the_file = open("data/" + current_scheme + "/varying_AAP/def_HEU.pkl", "wb+")
    pickle.dump(def_HEU_all_result, the_file)
    the_file.close()

    # Uncertainty
    the_file = open("data/" + current_scheme + "/varying_AAP/att_uncertainty.pkl",
                    "wb+")
    pickle.dump(att_uncertainty_all_result, the_file)
    the_file.close()
    the_file = open("data/" + current_scheme + "/varying_AAP/def_uncertainty.pkl",
                    "wb+")
    pickle.dump(def_uncertainty_all_result, the_file)
    the_file.close()

    # FPR & TPR
    the_file = open("data/" + current_scheme + "/varying_AAP/FPR.pkl", "wb+")
    pickle.dump(FPR_all_result, the_file)
    the_file.close()
    the_file = open("data/" + current_scheme + "/varying_AAP/TPR.pkl", "wb+")
    pickle.dump(TPR_all_result, the_file)
    the_file.close()
    the_file = open("data/" + current_scheme + "/varying_AAP/TP.pkl", "wb+")
    pickle.dump(TP_all_result, the_file)
    the_file.close()
    the_file = open("data/" + current_scheme + "/varying_AAP/FN.pkl", "wb+")
    pickle.dump(FN_all_result, the_file)
    the_file.close()
    the_file = open("data/" + current_scheme + "/varying_AAP/TN.pkl", "wb+")
    pickle.dump(TN_all_result, the_file)
    the_file.close()
    the_file = open("data/" + current_scheme + "/varying_AAP/FP.pkl", "wb+")
    pickle.dump(FP_all_result, the_file)
    the_file.close()


def run_sumulation_group_special(current_scheme, DD_using, uncertain_scheme_att, uncertain_scheme_def, decision_scheme, simulation_time, vary_name, vary_value):
    MTTSF_all_result = 0
    def_uncertainty_all_result = {}
    att_uncertainty_all_result = {}
    Time_to_SF_all_result = {}
    att_HEU_all_result = {}
    def_HEU_all_result = {}
    AHEU_per_Strategy_all_result = {}
    DHEU_per_Strategy_all_result = {}
    att_strategy_count_result = {}
    def_strategy_count_result = {}
    FPR_all_result = {}
    TPR_all_result = {}
    TP_all_result = {}
    FN_all_result = {}
    TN_all_result = {}
    FP_all_result = {}
    att_cost_all_result = {}
    def_cost_all_result = {}
    criticality_all_result = {}
    evict_reason_all_result = {}
    SysFail_reason = [0] * 3
    att_EU_C_all_result = {}
    att_EU_CMS_all_result = {}
    def_EU_C_all_result = {}
    def_EU_CMS_all_result = {}
    att_impact_all_result = {}
    def_impact_all_result = {}
    att_HEU_DD_IPI_all_result = {}
    def_HEU_DD_IPI_all_result = {}
    NIDS_eviction_all_result = {}
    number_of_attacker_all_result = {}
    att_CKC_all_result = {}
    compromise_probability_all_result = {}
    number_of_inside_attacker_all_result = {}
    all_result_after_each_game_all_result = {}

    results = []

    # "vary_value" and "vary_name" are used by ML to locate model
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for i in range(simulation_time):
            if vary_name == "AAP":
                future = executor.submit(game_start, i, DD_using,
                                         uncertain_scheme_att, uncertain_scheme_def,
                                         decision_scheme, current_scheme, att_arr_prob = vary_value,
                                         vary_name=vary_name, vary_value = vary_value)  # scheme change here
            else:
                future = executor.submit(
                    game_start, i, DD_using, uncertain_scheme_att, uncertain_scheme_def, decision_scheme,
                    current_scheme, web_data_upper_vul = vary_value, Iot_upper_vul = vary_value,
                    vary_name=vary_name, vary_value = vary_value)  # scheme change here
            results.append(future)

        index = 0

        for future in results:
            # MTTSF
            MTTSF_all_result += future.result().lifetime
            # New Attacker
            Time_to_SF_all_result[index] = future.result().lifetime
            # HEU
            att_HEU_all_result[index] = future.result().att_HEU_history
            def_HEU_all_result[index] = future.result().def_HEU_history
            # AHEU/DHEU per Strategy
            AHEU_per_Strategy_all_result[index] = future.result().AHEU_per_Strategy_History
            DHEU_per_Strategy_all_result[index] = future.result().DHEU_per_Strategy_History
            # Strategy Counter
            att_strategy_count_result[index] = future.result(
            ).att_strategy_counter
            def_strategy_count_result[index] = future.result(
            ).def_strategy_counter
            # Uncertainty
            def_uncertainty_all_result[index] = future.result(
            ).def_uncertainty_history
            att_uncertainty_all_result[index] = future.result(
            ).att_uncertainty_history
            # TPR & FPR
            FPR_all_result[index] = future.result().FPR_history
            TPR_all_result[index] = future.result().TPR_history
            # Cost
            att_cost_all_result[index] = future.result().att_cost_history
            def_cost_all_result[index] = future.result().def_cost_history
            # Criticality
            criticality_all_result[index] = future.result().criticality_hisotry
            # Evict attacker reason
            evict_reason_all_result[index] = future.result(
            ).evict_reason_history
            # System Fail reason
            if future.result().SysFail[0]:
                SysFail_reason[0] += 1  # [att_strat, system_fail]
            elif future.result().SysFail[1]:
                SysFail_reason[1] += 1
            elif future.result().SysFail[2]:
                SysFail_reason[2] += 1
            # EU_C & EU_CMS
            att_EU_C_all_result[index] = np.delete(future.result().att_EU_C, 0,
                                                   0)
            att_EU_CMS_all_result[index] = np.delete(
                future.result().att_EU_CMS, 0, 0)
            def_EU_C_all_result[index] = np.delete(future.result().def_EU_C, 0,
                                                   0)
            def_EU_CMS_all_result[index] = np.delete(
                future.result().def_EU_CMS, 0, 0)
            # attacker & defender impact
            att_impact_all_result[index] = np.delete(
                future.result().att_impact, 0, 0)
            def_impact_all_result[index] = np.delete(
                future.result().def_impact, 0, 0)
            # HEU in DD IPI
            att_HEU_DD_IPI_all_result[index] = np.delete(
                future.result().att_HEU_DD_IPI, 0, 0)
            def_HEU_DD_IPI_all_result[index] = np.delete(
                future.result().def_HEU_DD_IPI, 0, 0)
            # NIDS evict Bad or Good
            NIDS_eviction_all_result[index] = future.result().NIDS_eviction
            # Number of attacker
            number_of_attacker_all_result[index] = future.result().att_number
            # attacker CKC
            att_CKC_all_result[index] = future.result().att_CKC
            # compromised probability for AS2
            compromise_probability_all_result[index] = future.result().compromise_probability
            # inside attacker counter
            number_of_inside_attacker_all_result[index] = future.result().number_of_inside_attacker
            # data for ML model training
            all_result_after_each_game_all_result[index] = future.result().all_result_after_each_game


            index += 1

    # SAVE to FILE
    os.makedirs("data_vary/"+vary_name+"="+str(vary_value)+"/" + current_scheme + "/R0", exist_ok=True)
    the_file = open("data_vary/"+vary_name+"="+str(vary_value)+"/" + current_scheme + "/R0/Time_to_SF.pkl", "wb+")
    pickle.dump(Time_to_SF_all_result, the_file)
    the_file.close()

    # HEU
    os.makedirs("data_vary/"+vary_name+"="+str(vary_value)+"/" + current_scheme + "/R1", exist_ok=True)
    the_file = open("data_vary/"+vary_name+"="+str(vary_value)+"/" + current_scheme + "/R1/att_HEU.pkl", "wb+")
    pickle.dump(att_HEU_all_result, the_file)
    the_file.close()
    the_file = open("data_vary/"+vary_name+"="+str(vary_value)+"/" + current_scheme + "/R1/def_HEU.pkl", "wb+")
    pickle.dump(def_HEU_all_result, the_file)
    the_file.close()

    # Strategy Counter
    os.makedirs("data_vary/"+vary_name+"="+str(vary_value)+"/" + current_scheme + "/R2", exist_ok=True)
    the_file = open("data_vary/"+vary_name+"="+str(vary_value)+"/" + current_scheme + "/R2/att_strategy_counter.pkl",
                    "wb+")
    pickle.dump(att_strategy_count_result, the_file)
    the_file.close()
    the_file = open("data_vary/"+vary_name+"="+str(vary_value)+"/" + current_scheme + "/R2/def_strategy_counter.pkl",
                    "wb+")
    pickle.dump(def_strategy_count_result, the_file)
    the_file.close()

    # uncertainty
    os.makedirs("data_vary/"+vary_name+"="+str(vary_value)+"/" + current_scheme + "/R3", exist_ok=True)
    the_file = open("data_vary/"+vary_name+"="+str(vary_value)+"/" + current_scheme + "/R3/defender_uncertainty.pkl",
                    "wb+")
    pickle.dump(def_uncertainty_all_result, the_file)
    the_file.close()
    the_file = open("data_vary/"+vary_name+"="+str(vary_value)+"/" + current_scheme + "/R3/attacker_uncertainty.pkl",
                    "wb+")
    pickle.dump(att_uncertainty_all_result, the_file)
    the_file.close()

    # TPR & FPR
    os.makedirs("data_vary/"+vary_name+"="+str(vary_value)+"/" + current_scheme + "/R4", exist_ok=True)
    the_file = open("data_vary/"+vary_name+"="+str(vary_value)+"/" + current_scheme + "/R4/FPR.pkl", "wb+")
    pickle.dump(FPR_all_result, the_file)
    the_file.close()
    the_file = open("data_vary/"+vary_name+"="+str(vary_value)+"/" + current_scheme + "/R4/TPR.pkl", "wb+")
    pickle.dump(TPR_all_result, the_file)
    the_file.close()

    # MTTSF
    os.makedirs("data_vary/"+vary_name+"="+str(vary_value)+"/" + current_scheme + "/R5", exist_ok=True)
    the_file = open("data_vary/"+vary_name+"="+str(vary_value)+"/" + current_scheme + "/R5/MTTSF.pkl", "wb+")
    pickle.dump(MTTSF_all_result, the_file)
    the_file.close()

    # Cost
    os.makedirs("data_vary/"+vary_name+"="+str(vary_value)+"/" + current_scheme + "/R6", exist_ok=True)
    the_file = open("data_vary/"+vary_name+"="+str(vary_value)+"/" + current_scheme + "/R6/att_cost.pkl", "wb+")
    pickle.dump(att_cost_all_result, the_file)
    the_file.close()
    the_file = open("data_vary/"+vary_name+"="+str(vary_value)+"/" + current_scheme + "/R6/def_cost.pkl", "wb+")
    pickle.dump(def_cost_all_result, the_file)
    the_file.close()

    # Uncertainty
    os.makedirs("data_vary/"+vary_name+"="+str(vary_value)+"/" + current_scheme + "/R7", exist_ok=True)
    the_file = open("data_vary/"+vary_name+"="+str(vary_value)+"/" + current_scheme + "/R7/att_uncertainty.pkl",
                    "wb+")
    pickle.dump(att_uncertainty_all_result, the_file)
    the_file.close()
    the_file = open("data_vary/"+vary_name+"="+str(vary_value)+"/" + current_scheme + "/R7/def_uncertainty.pkl",
                    "wb+")
    pickle.dump(def_uncertainty_all_result, the_file)
    the_file.close()

    # attacker number
    os.makedirs("data_vary/"+vary_name+"="+str(vary_value)+"/" + current_scheme + "/R8", exist_ok=True)
    the_file = open("data_vary/"+vary_name+"="+str(vary_value)+"/" + current_scheme + "/R8/number_of_att.pkl",
                    "wb+")
    pickle.dump(number_of_attacker_all_result, the_file)
    the_file.close()

    # attacker CKC
    os.makedirs("data_vary/"+vary_name+"="+str(vary_value)+"/" + current_scheme + "/R9", exist_ok=True)
    the_file = open("data_vary/"+vary_name+"="+str(vary_value)+"/" + current_scheme + "/R9/att_CKC.pkl",
                    "wb+")
    pickle.dump(att_CKC_all_result, the_file)
    the_file.close()

    # ////////////////////# ////////////////////# ////////////////////

    # Criticality
    os.makedirs("data_vary/"+vary_name+"="+str(vary_value)+"/" + current_scheme + "/R_self_1", exist_ok=True)
    the_file = open("data_vary/"+vary_name+"="+str(vary_value)+"/" + current_scheme + "/R_self_1/criticality.pkl",
                    "wb+")
    pickle.dump(criticality_all_result, the_file)
    the_file.close()

    # Evict attacker reason
    os.makedirs("data_vary/"+vary_name+"="+str(vary_value)+"/" + current_scheme + "/R_self_2", exist_ok=True)
    the_file = open("data_vary/"+vary_name+"="+str(vary_value)+"/" + current_scheme + "/R_self_2/evict_reason.pkl",
                    "wb+")
    pickle.dump(evict_reason_all_result, the_file)
    the_file.close()

    # System Failure reason
    os.makedirs("data_vary/"+vary_name+"="+str(vary_value)+"/" + current_scheme + "/R_self_3", exist_ok=True)
    the_file = open("data_vary/"+vary_name+"="+str(vary_value)+"/" + current_scheme + "/R_self_3/system_fail.pkl",
                    "wb+")
    pickle.dump(SysFail_reason, the_file)
    the_file.close()

    # EU_C & EU_CMS
    os.makedirs("data_vary/"+vary_name+"="+str(vary_value)+"/" + current_scheme + "/R_self_4", exist_ok=True)
    the_file = open("data_vary/"+vary_name+"="+str(vary_value)+"/" + current_scheme + "/R_self_4/att_EU_C.pkl", "wb+")
    pickle.dump(att_EU_C_all_result, the_file)
    the_file.close()
    the_file = open("data_vary/"+vary_name+"="+str(vary_value)+"/" + current_scheme + "/R_self_4/att_EU_CMS.pkl",
                    "wb+")
    pickle.dump(att_EU_CMS_all_result, the_file)
    the_file.close()
    the_file = open("data_vary/"+vary_name+"="+str(vary_value)+"/" + current_scheme + "/R_self_4/def_EU_C.pkl", "wb+")
    pickle.dump(def_EU_C_all_result, the_file)
    the_file.close()
    the_file = open("data_vary/"+vary_name+"="+str(vary_value)+"/" + current_scheme + "/R_self_4/def_EU_CMS.pkl",
                    "wb+")
    pickle.dump(def_EU_CMS_all_result, the_file)
    the_file.close()

    # attacker & defender impact
    the_file = open("data_vary/"+vary_name+"="+str(vary_value)+"/" + current_scheme + "/R_self_4/att_impact.pkl",
                    "wb+")
    pickle.dump(att_impact_all_result, the_file)
    the_file.close()
    the_file = open("data_vary/"+vary_name+"="+str(vary_value)+"/" + current_scheme + "/R_self_4/def_impact.pkl",
                    "wb+")
    pickle.dump(def_impact_all_result, the_file)
    the_file.close()

    # HEU in DD IPI
    the_file = open("data_vary/"+vary_name+"="+str(vary_value)+"/" + current_scheme + "/R_self_4/att_HEU_DD_IPI.pkl",
                    "wb+")
    pickle.dump(att_HEU_DD_IPI_all_result, the_file)
    the_file.close()
    the_file = open("data_vary/"+vary_name+"="+str(vary_value)+"/" + current_scheme + "/R_self_4/def_HEU_DD_IPI.pkl",
                    "wb+")
    pickle.dump(def_HEU_DD_IPI_all_result, the_file)
    the_file.close()

    # NIDS evict good or bad
    the_file = open("data_vary/"+vary_name+"="+str(vary_value)+"/" + current_scheme + "/R_self_4/NIDS_eviction.pkl",
                    "wb+")
    pickle.dump(NIDS_eviction_all_result, the_file)
    the_file.close()

    # AHEU/DHEU per Strategy (HEU of all strategy)
    os.makedirs("data_vary/"+vary_name+"="+str(vary_value)+"/" + current_scheme + "/R_self_5", exist_ok=True)
    the_file = open("data_vary/"+vary_name+"="+str(vary_value)+"/" + current_scheme + "/R_self_5/AHEU_for_all_strategy_DD_IPI.pkl", "wb+")
    pickle.dump(AHEU_per_Strategy_all_result, the_file)
    the_file.close()
    the_file = open("data_vary/"+vary_name+"="+str(vary_value)+"/" + current_scheme + "/R_self_5/DHEU_for_all_strategy_DD_IPI.pkl", "wb+")
    pickle.dump(DHEU_per_Strategy_all_result, the_file)
    the_file.close()

    # compromised probability
    the_file = open("data_vary/"+vary_name+"="+str(vary_value)+"/" + current_scheme + "/R_self_5/compromise_probability_all_result.pkl", "wb+")
    pickle.dump(compromise_probability_all_result, the_file)
    the_file.close()

    # number of inside attacker
    the_file = open("data_vary/"+vary_name+"="+str(vary_value)+"/" + current_scheme + "/R_self_5/number_of_inside_attacker_all_result.pkl", "wb+")
    pickle.dump(number_of_inside_attacker_all_result, the_file)
    the_file.close()

    # training and target data for ML model
    os.makedirs("data_vary/"+vary_name+"="+str(vary_value)+"/trainning_data/" + current_scheme, exist_ok=True)
    # file_list = os.listdir("data/trainning_data/" + current_scheme)
    # file_ID = str(len(file_list) + 1)
    # the_file = open("data/" + current_scheme + "/ML_data/training/all_result_after_each_game_"+file_ID+".pkl", "wb+")
    now = datetime.now()
    dt_string = now.strftime("%H-%M_%d-%m-%Y")
    the_file = open("data_vary/"+vary_name+"="+str(vary_value)+"/trainning_data/" + current_scheme + "/all_result_after_each_game_" + dt_string + ".pkl",
                    "wb+")
    pickle.dump(all_result_after_each_game_all_result, the_file)
    the_file.close()

def run_sumulation_group_ML_data_collect(current_scheme, uncertain_scheme_att, uncertain_scheme_def, decision_scheme, simulation_time):
    DD_using = True

    ML_x_data_all_result = {}
    ML_y_data_all_result = {}

    results = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for i in range(simulation_time):
            future = executor.submit(game_start, i, DD_using,
                                     uncertain_scheme_att, uncertain_scheme_def, decision_scheme, current_scheme)  # scheme change here
            results.append(future)

        index = 0

        for future in results:
            ML_x_data_all_result[index] = future.result().ML_x_data
            ML_y_data_all_result[index] = future.result().ML_y_data

            index += 1

    # SAVE to FILE
    # training and target data for ML model
    os.makedirs("data/trainning_data/" + current_scheme, exist_ok=True)
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d-%H-%M")
    the_file = open("data/trainning_data/" + current_scheme + "/all_result_after_each_game_" + dt_string + ".pkl",
                    "wb+")
    pickle.dump([ML_x_data_all_result, ML_y_data_all_result], the_file)
    the_file.close()

def run_sumulation_group_ML_data_collect_vary_AAP(current_scheme, uncertain_scheme_att, uncertain_scheme_def, decision_scheme, simulation_time):
    DD_using = True
    AAP_list = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    vary_name = "AAP"

    for AAP in AAP_list:
        ML_x_data_all_result = {}
        ML_y_data_all_result = {}

        results = []

        with concurrent.futures.ProcessPoolExecutor() as executor:
            for i in range(simulation_time):
                future = executor.submit(game_start, i, DD_using,
                                         uncertain_scheme_att, uncertain_scheme_def, decision_scheme, current_scheme, att_arr_prob = AAP,
                                         vary_name=vary_name, vary_value = AAP)  # scheme change here
                results.append(future)

            index = 0

            for future in results:
                ML_x_data_all_result[index] = future.result().ML_x_data
                ML_y_data_all_result[index] = future.result().ML_y_data

                index += 1

        # SAVE to FILE
        # training and target data for ML model
        os.makedirs("data_vary/AAP=" + str(AAP) + "/trainning_data/" + current_scheme, exist_ok=True)
        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d-%H-%M")
        the_file = open("data_vary/AAP=" + str(AAP) + "/trainning_data/" + current_scheme + "/all_result_after_each_game_" + dt_string + ".pkl",
                        "wb+")
        pickle.dump([ML_x_data_all_result, ML_y_data_all_result], the_file)
        the_file.close()

def run_sumulation_group_ML_data_collect_vary_VUB(current_scheme, uncertain_scheme_att, uncertain_scheme_def, decision_scheme, simulation_time):
    DD_using = True
    VUB_list = np.array(range(1, 5 + 1)) * 2
    vary_name = "VUB"

    for VUB in VUB_list:
        ML_x_data_all_result = {}
        ML_y_data_all_result = {}

        results = []

        with concurrent.futures.ProcessPoolExecutor() as executor:
            for i in range(simulation_time):
                future = executor.submit(game_start, i, DD_using,
                                         uncertain_scheme_att, uncertain_scheme_def, decision_scheme,
                                         current_scheme, att_arr_prob=VUB,
                                         vary_name=vary_name, vary_value=VUB)  # scheme change here
                results.append(future)

            index = 0

            for future in results:
                ML_x_data_all_result[index] = future.result().ML_x_data
                ML_y_data_all_result[index] = future.result().ML_y_data

                index += 1

        # SAVE to FILE
        # training and target data for ML model
        os.makedirs("data_vary/VUB=" + str(VUB) + "/trainning_data/" + current_scheme, exist_ok=True)
        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d-%H-%M")
        the_file = open("data_vary/VUB=" + str(
            VUB) + "/trainning_data/" + current_scheme + "/all_result_after_each_game_" + dt_string + ".pkl",
                        "wb+")
        pickle.dump([ML_x_data_all_result, ML_y_data_all_result], the_file)
        the_file.close()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    start = time.time()
    simulation_time = 1000
    print(f"number of core: {multiprocessing.cpu_count()}")
    game_start()
    # decision_scheme: 0 means random, 1 means HEU, 2 means ML, 3 means ML Data Collection
    # (current_scheme, DD_using, uncertain_scheme_att, uncertain_scheme_def, decision_scheme, simulation_time)

    # Static Vulnerability
    # run_sumulation_group_1("DD-Random", True, True, True, 0, simulation_time)
    # run_sumulation_group_1("No-DD-Random", False, True, True, 0, simulation_time)
    # run_sumulation_group_1("DD-IPI", True, True, True, 1, simulation_time)
    # run_sumulation_group_1("DD-ML-IPI", True, True, True, 2, simulation_time)
    # run_sumulation_group_1("No-DD-IPI", False, True, True, 1, simulation_time)
    # run_sumulation_group_1("DD-PI", True, False, False, 1, simulation_time)
    # run_sumulation_group_1("DD-ML-PI", True, False, False, 2, simulation_time)
    # run_sumulation_group_1("No-DD-PI", False, False, False, 1, simulation_time)
    # run_sumulation_group_1("DD-IPI_ML_data", True, True, False, 1, simulation_time)

    # Varying Vulnerability Upper Bound (VUB)
    # run_sumulation_group_vary_VUB("DD-Random", True, True, True, 0, simulation_time)
    # run_sumulation_group_vary_VUB("No-DD-Random", False, True, True, 0, simulation_time)
    # run_sumulation_group_vary_VUB("DD-IPI", True, True, True, 1, simulation_time)
    # run_sumulation_group_vary_VUB("DD-ML-IPI", True, True, True, 2, simulation_time)
    # run_sumulation_group_vary_VUB("No-DD-IPI", False, True, True, 1, simulation_time)
    # run_sumulation_group_vary_VUB("DD-PI", True, False, False, 1, simulation_time)
    # run_sumulation_group_vary_VUB("DD-ML-PI", True, False, False, 2, simulation_time)
    # run_sumulation_group_vary_VUB("No-DD-PI", False, False, False, 1, simulation_time)
    # run_sumulation_group_vary_VUB("DD-IPI_ML_data", True, True, False, 1, simulation_time)

    # Varying Attacker Arrival Probability (AAP)
    # run_sumulation_group_vary_AAP("DD-Random", True, True, True, 0, simulation_time)
    # run_sumulation_group_vary_AAP("No-DD-Random", False, True, True, 0, simulation_time)
    # run_sumulation_group_vary_AAP("DD-IPI", True, True, True, 1, simulation_time)
    # run_sumulation_group_vary_AAP("DD-ML-IPI", True, True, True, 2, simulation_time)
    # run_sumulation_group_vary_AAP("No-DD-IPI", False, True, True, 1, simulation_time)
    # run_sumulation_group_vary_AAP("DD-PI", True, False, False, 1, simulation_time)
    # run_sumulation_group_vary_AAP("DD-ML-PI", True, False, False, 2, simulation_time)
    # run_sumulation_group_vary_AAP("No-DD-PI", False, False, False, 1, simulation_time)
    run_sumulation_group_vary_AAP("DD-IPI_ML_data", True, True, False, 1, simulation_time)

    # Collect data for ML
    # run_sumulation_group_ML_data_collect("ML_collect_data_PI", False, False, 3, simulation_time)
    # run_sumulation_group_ML_data_collect("ML_collect_data_IPI", True, True, 3, simulation_time)
    # run_sumulation_group_ML_data_collect_vary_AAP("ML_collect_data_PI", False, False, 3, simulation_time)
    # run_sumulation_group_ML_data_collect_vary_AAP("ML_collect_data_IPI", True, True, 3, simulation_time)
    # run_sumulation_group_ML_data_collect_vary_VUB("ML_collect_data_PI", False, False, 3, simulation_time)
    # run_sumulation_group_ML_data_collect_vary_VUB("ML_collect_data_IPI", True, True, 3, simulation_time)


# ============================================================
    # Save data for each setting
    AAP_list = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    VUB_list = np.array(range(1, 5 + 1)) * 2

    # for AAP in AAP_list:
        # run_sumulation_group_special("DD-Random", True, True, True, 0, simulation_time, "AAP", AAP)
        # run_sumulation_group_special("No-DD-Random", False, True, True, 0, simulation_time, "AAP", AAP)
        # run_sumulation_group_special("DD-IPI", True, True, True, 1, simulation_time, "AAP", AAP)
        # run_sumulation_group_special("DD-ML-IPI", True, True, True, 2, simulation_time, "AAP", AAP)
        # run_sumulation_group_special("No-DD-IPI", False, True, True, 1, simulation_time, "AAP", AAP)
        # run_sumulation_group_special("DD-PI", True, False, False, 1, simulation_time, "AAP", AAP)
        # run_sumulation_group_special("DD-ML-PI", True, False, False, 2, simulation_time, "AAP", AAP)
        # run_sumulation_group_special("No-DD-PI", False, False, False, 1, simulation_time, "AAP", AAP)
        # run_sumulation_group_special("DD-IPI_ML_data", True, True, False, 1, simulation_time, "AAP", AAP)

    # for VUB in VUB_list:
        # run_sumulation_group_special("DD-Random", True, True, True, 0, simulation_time, "VUB", VUB)
        # run_sumulation_group_special("No-DD-Random", False, True, True, 0, simulation_time, "VUB", VUB)
        # run_sumulation_group_special("DD-IPI", True, True, True, 1, simulation_time, "VUB", VUB)
        # run_sumulation_group_special("DD-ML-IPI", True, True, True, 2, simulation_time, "VUB", VUB)
        # run_sumulation_group_special("No-DD-IPI", False, True, True, 1, simulation_time, "VUB", VUB)
        # run_sumulation_group_special("DD-PI", True, False, False, 1, simulation_time, "VUB", VUB)
        # run_sumulation_group_special("DD-ML-PI", True, False, False, 2, simulation_time, "VUB", VUB)
        # run_sumulation_group_special("No-DD-PI", False, False, False, 1, simulation_time, "VUB", VUB)
        # run_sumulation_group_special("DD-IPI_ML_data", True, True, False, 1, simulation_time, "VUB", VUB)

    print("Project took", time.time() - start, "seconds.")
    try:
        os.system('say "your program has finished"')
        os._exit(0)
    except:
        print("command not found: say")

    # os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))

    # run_sumulation_group_1("DD-Random-PI", True, False, 0, simulation_time) # removed
