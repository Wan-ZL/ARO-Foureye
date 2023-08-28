#!/bin/bash

#SBATCH --ntasks=300
#SBATCH -t 00-10:00:00
#SBATCH -p normal_q	# or dev_q
#SBATCH --account=zelin
#SBATCH --export=NONE # this makes sure the compute environment is clean

echo "TinkerCliffs Start"

echo "Core Number:"
nproc --all

pwd
ls

echo "load Anaconda"
module load Anaconda3

conda list -f scikit-learn


python --version

python main.py


echo "Scrpt End"