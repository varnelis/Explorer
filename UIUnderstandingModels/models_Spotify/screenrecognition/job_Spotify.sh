#!/bin/bash

#SBATCH --nodes=1
#SBATCH --job-name=L16LR004-ResMax
#SBATCH --gres=gpu:8
#SBATCH --partition=big
#SBATCH --time=01-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=iason.chaimalas.20@ucl.ac.uk

module load python/miniconda3

source activate webui

srun python ui_train_spotify.py 8 32 16 0.04 -1 720 1560 &> ./slurm-spotify/slurm-$SLURM_JOB_ID.out
# gpu, prec, batch, lr, subsize, minsize, maxsize
