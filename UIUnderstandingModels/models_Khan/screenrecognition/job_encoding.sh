#!/bin/bash

#SBATCH --nodes=1
#SBATCH --job-name=all_encode_compare
        # currently only using the data I scraped
#SBATCH --gres=gpu:8
#SBATCH --partition=big
#SBATCH --time=00-10:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=iason.chaimalas.20@ucl.ac.uk

module load python/anaconda3

source activate webui

python encoder_exploration.py
