#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=00:00:30
#SBATCH --job-name=helloworldtest
#SBATCH --gres=gpu:1
#SBATCH --partition=devel
#SBATCH --mail-type=ALL
#SBATCH --mail-user=iason.chaimalas.20@ucl.ac.uk

module load python/3.8.6

python hello_world.py