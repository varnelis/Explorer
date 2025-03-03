#!/bin/bash

#SBATCH --nodes=1
#SBATCH --job-name=test
#SBATCH --gres=gpu:1
#SBATCH --partition=devel
#SBATCH --time=00-01:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=iason.chaimalas.20@ucl.ac.uk

module load python/anaconda3

source activate webui

python khaninteractable_dataloading_test_script.py
