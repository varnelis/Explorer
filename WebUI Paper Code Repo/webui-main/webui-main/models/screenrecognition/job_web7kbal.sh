#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --job-name=web7kbal
#SBATCH --gres=gpu:1
#SBATCH --partition=devel
#SBATCH --mail-type=ALL
#SBATCH --mail-user=iason.chaimalas.20@ucl.ac.uk

module load python/anaconda3

source activate webui

python ui_train_web7kbal.py
