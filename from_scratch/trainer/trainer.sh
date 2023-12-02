#!/bin/bash

#SBATCH -p gpu --gres=gpu:2
#SBATCH -n 2
#SBATCH --mem=150GB
#SBATCH -t 12:00:00

#SBATCH -J icetrainer

#SBATCH -o trainer-%j.out
#SBATCH -e trainer-%j.out

module load python/3.9.0
module load gcc/10.2
module load cuda/11.3.1

source /users/wwoodber/data/icelandic-nlp/env/bin/activate

python3 /users/wwoodber/data/icelandic-nlp/from_scratch/trainer/trainer.py