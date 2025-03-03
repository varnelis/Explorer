#!/bin/bash

#SBATCH --nodes=1
#SBATCH --job-name=L16LR004-ResMax-Wikipedia-web7kfinetune-notrain
#SBATCH --gres=gpu:8
#SBATCH --partition=big
#SBATCH --time=00-08:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=iason.chaimalas.20@ucl.ac.uk

module load python/miniconda3

source activate webui

srun python ui_train_wikipedia_web7kfinetune.py 8 32 4 0.01 -1 1600 2400 &> ./slurm-wikipedia/slurm-$SLURM_JOB_ID.out
# gpu, prec, batch, lr, subsize, minsize, maxsize
