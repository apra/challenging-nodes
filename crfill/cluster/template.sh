#!/bin/bash
#SBATCH -t 10:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=samuele.papa@gmail.com
#SBATCH -p gpu_titanrtx
#SBATCH --mem=75G
#SBATCH --gpus-per-node=titanrtx:4
#SBATCH --cpus-per-task=12
#SBATCH -N 1
#SBATCH --output=experiment_name%A.out
#SBATCH --error=experiment_name%A.err

DATA="$TMPDIR/spapa"

#Create output directory on scratch
mkdir -p "$DATA"
cp -ra /project/spapa/dataset_node21/. "$DATA"