#!/bin/bash
#SBATCH -t 02:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=samuele.papa@gmail.com
#SBATCH -p gpu_shared
#SBATCH --gpus-per-node=gtx1080ti:1
#SBATCH -N 1
#SBATCH --output=array_%A.out
#SBATCH --error=array_%A.err

#Loading modules
module load 2021
module load Python/3.9.5-GCCcore-10.3.0

#Create output directory on scratch
mkdir -p "$TMP_DIR/spapa"
cp -r $HOME/data/ "$TMP_DIR/spapa"
