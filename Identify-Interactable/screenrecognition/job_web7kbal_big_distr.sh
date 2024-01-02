#!/bin/bash

#SBATCH --nodes=1
#SBATCH --job-name=web7kbalBIG
#SBATCH --gres=gpu:8
#SBATCH --partition=big
#SBATCH --mail-type=ALL
#SBATCH --mail-user=iason.chaimalas.20@ucl.ac.uk

module load python/anaconda3

source activate webui

python ui_train_web7kbal_distributed.py
