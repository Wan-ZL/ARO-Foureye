#!/bin/bash

echo "load environment: Drone-NetworkX-PyTorch"
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate Drone-NetworkX-PyTorch

def_select_method_list=('CD' 'CD' 'CD') #('fixed' 'HEU' 'DRL' 'HT-DRL' 'No-Defense' 'IDS' 'CD')
att_select_method_list=('fixed' 'HEU' 'DRL' 'HT-DRL') #('fixed' 'HEU' 'DRL' 'HT-DRL')
miss_dur_list=(150) #(110 130 150 170 190)
miss_dur_default=150
att_budget_list=(1 3 7 9)
att_budget_default=5
test_mode_run_time=1

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


