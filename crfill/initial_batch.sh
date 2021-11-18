#!/bin/bash
#SBATCH -t 04:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=samuele.papa@gmail.com
#SBATCH -p gpu_short
#SBATCH -N 1

cp -r $HOME/data/ $TMP_DIR

