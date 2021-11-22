#!/bin/bash
#SBATCH -t 02:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=samuele.papa@gmail.com
#SBATCH -p gpu_shared
#SBATCH --gpus-per-node=gtx1080ti:1
#SBATCH -N 1
#SBATCH --output=test_%A.out
#SBATCH --error=test_%A.err

#Loading modules
module load 2021
module load Python/3.9.5-GCCcore-10.3.0
module load PyTorch/1.10.0-foss-2021a-CUDA-11.3.1

#Create output directory on scratch
mkdir -p "$TMPDIR/spapa"
cp -r $HOME/data/ "$TMPDIR/spapa"
