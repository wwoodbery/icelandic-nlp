#!/bin/bash

#SBATCH -n 2
#SBATCH --mem=20GB
#SBATCH -t 00:10:00

#SBATCH -J bert_example

#SBATCH -o bert-%j.out
#SBATCH -e bert-%j.out

module load python/3.9.0
# module load gcc/10.2
# module load cuda/11.3.1

source /users/wwoodber/data/wwoodber/project-icelandic/env/bin/activate && python3 /users/wwoodber/data/wwoodber/project-icelandic/bert_example.py
echo "Python Version: $(which python)"
echo "Virtual Environment: $VIRTUAL_ENV"

python3 /users/wwoodber/data/wwoodber/project-icelandic/bert_example.py