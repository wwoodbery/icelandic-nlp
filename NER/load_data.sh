#!/bin/bash

#SBATCH -n 2
#SBATCH --mem=20GB
#SBATCH -t 00:02:00

#SBATCH -J icebert

#SBATCH -o ner_load_data-%j.out
#SBATCH -e ner_load_data-%j.out

module load python/3.9.0
# module load gcc/10.2
# module load cuda/11.3.1

source /users/wwoodber/data/icelandic-nlp/env/bin/activate

python3 /users/wwoodber/data/icelandic-nlp/NER/load_data.py