#!/bin/bash

#SBATCH --nodes=1
#SBATCH --job-name=KhanInteractableDetection
#SBATCH --gres=gpu:8
#SBATCH --partition=big
#SBATCH --time=01-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=iason.chaimalas.20@ucl.ac.uk

module load python/anaconda3

source activate webui

srun python ui_train_finetune_khan.py 8 32 32 0.08
# gpu, prec, batch, lr

# Training over 8 GPUs in JADE SLURM cluster
# Equivalent 1GPU Batch 256, LR 0.64