#!/bin/bash
#SBATCH -t 02:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=samuele.papa@gmail.com
#SBATCH -p gpu_shared
#SBATCH --gpus-per-node=gtx1080ti:2
#SBATCH -N 1
#SBATCH --output=test_%A.out
#SBATCH --error=test_%A.err

JOBS_SOURCE="$HOME/challenging-nodes/crfill"
SINGULARITYIMAGE=$HOME/image.sif

run_train()
{
  echo "python -u docker_folder/test_cuda_pytorch.py"
}
COMMAND=$(run_train)
slurm_submit()
{
  singularity exec --no-home --nv \
    --bind "$JOBS_SOURCE" \
    --pwd "$JOBS_SOURCE" \
    $SINGULARITYIMAGE \
    $COMMAND &
}
echo $(slurm_submit $COMMAND)