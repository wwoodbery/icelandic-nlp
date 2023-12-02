#!/bin/bash

#SBATCH -n 1
#SBATCH --mem=1GB
#SBATCH -t 00:01:00

#SBATCH -J hello_world

#SBATCH -o hello-%j.out
#SBATCH -e hello-%j.out

module load python/3.9.0

source /users/wwoodber/data/wwoodber/project-icelandic/env/bin/activate

python3 /users/wwoodber/data/wwoodber/project-icelandic/hello_world.py