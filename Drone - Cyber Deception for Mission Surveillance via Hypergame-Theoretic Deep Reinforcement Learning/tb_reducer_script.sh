#!/bin/bash

# =========== for local Mac ===========
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate Drone-NetworkX-PyTorch
path=$(pwd)"/data"

# =========== for ARC linux ============
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=18
#SBATCH -t 00-01:00:00
#SBATCH -p normal_q
#SBATCH --account=zelin1
#SBATCH --export=NONE # this makes sure the compute environment is clean
# uncomment below for ARC linux
#module load Anaconda3
#source activate Drone-NetworkX-PyTorch
#path="/projects/zelin1/Drone_DRL_HT/data"
# ======================================

echo "load environment: Drone-NetworkX-PyTorch"

simu_setting_list=(150_5_0) #(110_5_5 130_5_5 150_1_5 150_3_5 150_5_5 150_7_5 150_9_5 170_5_5 190_5_5)
def_select_method_list=('No-Defense' 'IDS' 'CD') #('fixed' 'HEU' 'DRL' 'HT-DRL')
att_select_method_list=('fixed' 'HEU' 'DRL' 'HT-DRL')
#scheme_list=(scheme_DefAtt) #(scheme_att scheme_def scheme_DefAtt scheme_random)
#HEU_list=(HEU_def=False_att=False HEU_def=True_att=False HEU_def=False_att=True HEU_def=True_att=True)
#entropy_threshold_list=(0.005 0.01 0.05 0.1 0.2)
#epsilon_list=(0.4 0.6 0.8 1.0) #(0.2 0.4 0.6 0.8 1.0)

#simu_setting_list=(150_5_5)
#scheme_list=(scheme_att)
#HEU_list=(HEU_def=False_att=False)

# run processes and store pids in array
for simu_setting in "${simu_setting_list[@]}"
do
    pid_counter=0
    for att_select_method in "${att_select_method_list[@]}"
    do
        for def_select_method in "${def_select_method_list[@]}"
        do
            tb-reducer $path/$simu_setting/scheme_${att_select_method}-${def_select_method}/* -o $path/tb_reduce/$simu_setting/scheme_${att_select_method}-${def_select_method}/ -r mean --handle-dup-steps 'mean' --lax-step --lax-tags &
            # tb-reducer $path/$simu_setting/entropy_threshold_${entropy_threshold}_epsilon_$epsilon/$scheme/$HEU/* -o $path/tb_reduce/$simu_setting/entropy_threshold_${entropy_threshold}_epsilon_$epsilon/$scheme/$HEU/ -r mean --handle-dup-steps 'mean' --lax-step --lax-tags &
            pids[${pid_counter}]=$!
            pid_counter=$((pid_counter+1))
            echo "pids: ${pids[@]}"
        done
    done
    # wait for all pids
    for pid in ${pids[*]}; do
        wait $pid
    done
done

echo "tb_reducer_script.sh finished"