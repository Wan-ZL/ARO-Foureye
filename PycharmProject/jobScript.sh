#!/bin/bash

#SBATCH --ntasks=32
#SBATCH -t 00-03:00:00


echo "Scrpt Start"

echo "Core Number:"
nproc --all

pwd
ls

echo "load Anaconda"
module load Anaconda


python --version

python Foureye_parallel_simulation.py


echo "Scrpt End"