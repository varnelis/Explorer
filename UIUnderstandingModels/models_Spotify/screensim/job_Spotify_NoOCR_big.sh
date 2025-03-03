#!/bin/bash

#SBATCH --nodes=1
#SBATCH --job-name=L_LR0256P16B16_screensim-big-spotify
#SBATCH --gres=gpu:8
#SBATCH --partition=big
#SBATCH --time=01-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=iason.chaimalas.20@ucl.ac.uk

module load python/miniconda3

source activate webui

#srun python ui_train_spotify_noocr_plus_centerness.py 8 16 0.0256 0 16
srun python ui_train_spotify_noocr_plus_centerness_EVAL.py 8 16 0.0256 0 16
# gpu, prec, lr, lambda_ocr, batch_size
