#!/bin/bash

#SBATCH --ntasks=100
#SBATCH -t 00-05:00:00
#SBATCH -p normal_q	# normal_q or dev_q
#SBATCH --account=zelin
#SBATCH --export=NONE # this makes sure the compute environment is clean

echo "TinkerCliffs Start"

echo "Core Number:"
nproc --all

pwd
ls

cat main.py

echo "load Anaconda"
module load Anaconda3

conda list -f scikit-learn


python --version

python main.py


echo "Scrpt End"