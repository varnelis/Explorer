#!/bin/bash

#SBATCH --nodes=1
#SBATCH --job-name=L_norelu_LR0128P16B64_screensim-big
#SBATCH --gres=gpu:8
#SBATCH --partition=big
#SBATCH --time=01-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=iason.chaimalas.20@ucl.ac.uk

module load python/anaconda3

source activate webui

srun python ui_train_khan_noocr_noReLU.py 8 16 0.0128 0 64
# gpu, prec, lr, lambda_ocr, batch_size
