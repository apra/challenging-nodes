#!/bin/bash
#SBATCH -t 02:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=samuele.papa@gmail.com
#SBATCH -p gpu_short
#SBATCH -N 1

#Loading modules
module load 2021
module load Python/3.9.5-GCCcore-10.3.0

#Create output directory on scratch
cp -r $HOME/data/ $TMP_DIR
