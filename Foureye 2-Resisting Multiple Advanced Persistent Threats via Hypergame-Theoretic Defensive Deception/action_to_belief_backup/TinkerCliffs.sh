#!/bin/bash

#SBATCH --ntasks=32
#SBATCH -t 00-02:00:00
#SBATCH -p normal_q
#SBATCH --account=personal
#SBATCH --export=NONE # this makes sure the compute environment is clean

echo "TinkerCliffs Start"

echo "Core Number:"
nproc --all

pwd
ls

echo "load Anaconda"
module load Anaconda

conda list -f scikit-learn


python --version

python main.py


echo "Scrpt End"