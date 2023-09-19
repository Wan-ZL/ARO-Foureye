#!/bin/bash

#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=100
#SBATCH -t 00-06:00:00
#SBATCH -p normal_q
#SBATCH --account=zelin1
#SBATCH --export=NONE # this makes sure the compute environment is clean

echo "TinkerCliffs Start"

echo "Core Number:"
nproc --all

pwd
ls

echo "load Anaconda"
module load Anaconda3

echo "activate environment: dronePytorch"
source activate Drone-NetworkX-PyTorch

python --version

echo "set OMP_NUM_THREADS=1"
export OMP_NUM_THREADS=1

# ==================================================================================================
exp_scheme_list=(2) #(0 1 2 3)
HEU_def_list=(True False)
HEU_att_list=(True False)
miss_dur_list=(150) #(110 130 150 170 190)
att_budget_list=(5) # (1 3 5 7 9)
test_mode_run_time=4
entropy_threshold_list=(0.005 0.01 0.05 0.1 0.2) #(0.01 0.05 0.1 0.2 0.4 0.6)
epsilon_list=(0.4 0.6 0.8 1.0)

# run processes and store pids in array
pid_counter=0
for exp_scheme in "${exp_scheme_list[@]}"
do
    for HEU_def in "${HEU_def_list[@]}"
    do
        for HEU_att in "${HEU_att_list[@]}"
        do
            for miss_dur in "${miss_dur_list[@]}"
            do
                for att_budget in "${att_budget_list[@]}"
                do
                    for entropy_threshold in "${entropy_threshold_list[@]}"
                    do
                        for epsilon in "${epsilon_list[@]}"
                        do
                            echo "exp_scheme: $exp_scheme, HEU_def: $HEU_def, HEU_att: $HEU_att, miss_dur: $miss_dur, att_budget: $att_budget, test_mode_run_time: $test_mode_run_time, entropy_threshold: $entropy_threshold, epsilon: $epsilon"
                            python3 Optuna_A3C_1.py --exp_scheme $exp_scheme --HEU_def $HEU_def --HEU_att $HEU_att --miss_dur $miss_dur --max_att_budget $att_budget --run_time $test_mode_run_time --entropy_threshold $entropy_threshold --epsilon $epsilon &
                            pids[${pid_counter}]=$!
                            pid_counter=$((pid_counter+1))
                            echo "pids: ${pids[@]}"
                        done
                    done
                done
            done
        done
    done
done

# wait for all pids
for pid in ${pids[*]}; do
    wait $pid
done
# ==================================================================================================

echo "Script End"