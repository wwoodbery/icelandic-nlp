#!/bin/bash

#SBATCH -p gpu --gres=gpu:2
#SBATCH -n 2
#SBATCH --mem=40GB
#SBATCH -t 01:00:00

#SBATCH -J icebert

#SBATCH -o ner-%j.out
#SBATCH -e ner-%j.out

module load python/3.9.0
# module load gcc/10.2
# module load cuda/11.3.1

source /users/wwoodber/data/icelandic-nlp/env/bin/activate

python3 /users/wwoodber/data/icelandic-nlp/NER/ner.py