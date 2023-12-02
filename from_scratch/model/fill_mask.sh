#!/bin/bash

#SBATCH -n 2
#SBATCH --mem=20GB
#SBATCH -t 00:10:00

#SBATCH -J fill_mask

#SBATCH -o fill_mask-%j.out
#SBATCH -e fill_mask-%j.out

module load python/3.9.0

source /users/wwoodber/data/wwoodber/project-icelandic/env/bin/activate

python3 /users/wwoodber/data/icelandic-nlp/from_scratch/model/fill_mask.py