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
DATA="$TMPDIR/spapa"

LOGGING_DIR="$HOME/challenging-nodes/crfill/checkpoints"

#Create output directory on scratch
mkdir -p "$DATA"
cp -r $HOME/data/ "$DATA"
run_train()
{
  echo "python -u train.py \
        --dataset_mode_train \
        custom_train \
        --name \
        debug \
        --checkpoints_dir \
        checkpoints/1 \
        --dataset_mode \
        custom_train \
        --train_image_dir \
        /data/images/cxr_images/proccessed_data/images \
        --train_nodule_list \
        /data/images/cxr_images/proccessed_data/metadata.csv \
        --netG \
        twostagend \
        --netD \
        deepfill \
        --preprocess_mode \
        none \
        --validation_freq \
        100 \
        --gpu_ids \
        0,1 \
        --niter \
        50 \
        --batchSize \
        64 \
        --display_freq \
        20 \
        --model \
        arrange"
}
COMMAND=$(run_train)

slurm_submit()
{
singularity exec --no-home --nv \
    --bind "$DATA":/data \
    --bind $LOGGING_DIR:$LOGGING_DIR \
    --bind $DATA_ROOT \
    --bind "$JOBS_SOURCE" \
    --bind "$DATA" \
    --pwd "$JOBS_SOURCE" \
    $SINGULARITYIMAGE \
    $COMMAND &
}

echo $(slurm_submit $COMMAND)