#!/bin/bash

#SBATCH --nodes=1
#SBATCH --job-name=eval0-L_LR0032P16B16_screensim-small-spotify
#SBATCH --gres=gpu:1
#SBATCH --partition=small
#SBATCH --time=01-12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=iason.chaimalas.20@ucl.ac.uk

module load python/miniconda3

source activate webui

#srun python ui_train_spotify_noocr_plus_centerness.py 1 16 0.0032 0 16
srun python ui_train_spotify_noocr_plus_centerness_EVAL.py 1 16 0.0032 0 16
# gpu, prec, lr, lambda_ocr, batch_size
