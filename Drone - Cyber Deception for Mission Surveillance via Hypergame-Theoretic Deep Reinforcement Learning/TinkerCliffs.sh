#!/bin/bash

#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=93
#SBATCH -t 00-12:00:00
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
def_select_method_list=('fixed' 'HEU' 'DRL' 'No-Defense' 'IDS' 'CD') #('fixed' 'HEU' 'DRL' 'HT-DRL' 'No-Defense' 'IDS' 'CD')
att_select_method_list=('fixed' 'HEU' 'DRL')
miss_dur_list=(150) #(110 130 150 170 190)
miss_dur_default=150
att_budget_list=(1 3 7 9)
att_budget_default=5
test_mode_run_time=5

# varying miss_dur_list
pid_counter=0
for miss_dur in "${miss_dur_list[@]}"
do
  for def_select_method in "${def_select_method_list[@]}"
  do
    for att_select_method in "${att_select_method_list[@]}"
    do
      echo "def_select_method: $def_select_method, att_select_method: $att_select_method, miss_dur: $miss_dur, att_budget: $att_budget_default, test_mode_run_time: $test_mode_run_time"
      python3 Optuna_A3C_1.py --def_select_method $def_select_method --att_select_method $att_select_method --miss_dur $miss_dur --max_att_budget $att_budget_default --run_time $test_mode_run_time &
      pids[${pid_counter}]=$!
      pid_counter=$((pid_counter+1))
      echo "pids: ${pids[@]}"
    done
  done
done
# wait for all pids
for pid in ${pids[*]}; do
    wait $pid
done

# varying att_budget_list
#pid_counter=0
#for att_budget in "${att_budget_list[@]}"
#do
#  for def_select_method in "${def_select_method_list[@]}"
#  do
#    for att_select_method in "${att_select_method_list[@]}"
#    do
#      echo "def_select_method: $def_select_method, att_select_method: $att_select_method, miss_dur: $miss_dur_default, att_budget: $att_budget, test_mode_run_time: $test_mode_run_time"
#      python3 Optuna_A3C_1.py --def_select_method $def_select_method --att_select_method $att_select_method --miss_dur $miss_dur_default --max_att_budget $att_budget --run_time $test_mode_run_time &
#      pids[${pid_counter}]=$!
#      pid_counter=$((pid_counter+1))
#      echo "pids: ${pids[@]}"
#    done
#  done
#done
## wait for all pids
#for pid in ${pids[*]}; do
#    wait $pid
#done
# ==================================================================================================

echo "Script End"