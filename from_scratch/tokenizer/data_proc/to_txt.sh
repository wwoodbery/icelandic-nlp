#!/bin/bash

#SBATCH -p gpu --gres=gpu:2
#SBATCH -n 2
#SBATCH --mem=40GB
#SBATCH -t 01:00:00

#SBATCH -J to_txt

#SBATCH -o to_txt-%j.out
#SBATCH -e to_txt-%j.out

module load python/3.9.0
module load gcc/10.2
module load cuda/11.3.1

source /users/wwoodber/data/icelandic-nlp/env/bin/activate

python3 /users/wwoodber/data/icelandic-nlp/from_scratch/tokenizer/to_txt.py