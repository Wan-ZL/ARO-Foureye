#!/bin/bash

#SBATCH --ntasks=2
#SBATCH -t 00-00:05:00
#SBATCH -p a100_dev_q
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
source activate dronePytorch

python --version

python A3C_optuna.py


echo "Scrpt End"