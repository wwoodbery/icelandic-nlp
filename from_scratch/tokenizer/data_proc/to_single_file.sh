#!/bin/bash

#SBATCH -n 4
#SBATCH --mem=40GB
#SBATCH -t 00:30:00

#SBATCH -J to_single_file

#SBATCH -o to_single_file-%j.out
#SBATCH -e to_single_file-%j.out

module load python/3.9.0
# module load gcc/10.2
# module load cuda/11.3.1

source /users/wwoodber/data/icelandic-nlp/env/bin/activate

python3 /users/wwoodber/data/icelandic-nlp/from_scratch/tokenizer/data_proc/to_single_file.py