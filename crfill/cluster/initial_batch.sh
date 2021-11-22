#!/bin/bash
#SBATCH -t 02:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=samuele.papa@gmail.com
#SBATCH -p gpu_shared
#SBATCH --gpus-per-node=gtx1080ti:1
#SBATCH -N 1
#SBATCH --output=test_%A.out
#SBATCH --error=test_%A.err

JOBS_SOURCE=$HOME/crfill
SINGULARITYIMAGE=$HOME/crfill.sif
DATA="$TMPDIR/spapa"

#Create output directory on scratch
mkdir -p "$TMPDIR/spapa"
cp -r $HOME/data/ "$DATA"

singularity exec --no-home --nv \
    --bind "$DATA":/data \
    --bind $EXPERIMENT_DIR:$EXPERIMENT_DIR_CONTAINER \
    --bind $DATA_ROOT \
    --bind $MASKS_ROOT_DIR \
    --bind "$JOBS_SOURCE" \
    --bind "$TMPDIR" \
    --pwd /hissl \
    $SINGULARITYIMAGE \
    $COMMAND &