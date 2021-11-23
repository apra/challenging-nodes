#!/bin/bash

ROOT="$HOME/challenge"

JOBS_SOURCE="$ROOT"
SINGULARITYIMAGE="$ROOT/container.sif"
DATA="/mnt/teuwen/archive/data/radiology/Challenge_NODE21"

LOGGING_DIR="$ROOT/checkpoints"

run_train()
{
  echo "python -u train.py \
        --dataset_mode_train \
        custom_train \
        --name \
        debug \
        --checkpoints_dir \
        /log/checkpoints/1 \
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
echo "Before slurm_submit"
slurm_submit()
{
  singularity exec --no-home --nv \
    --bind "$DATA":/data \
    --bind $LOGGING_DIR:"/log/$LOGGING_DIR" \
    --bind "$JOBS_SOURCE" \
    --bind "$DATA" \
    --pwd "$JOBS_SOURCE" \
    $SINGULARITYIMAGE \
    $COMMAND &
}
echo "Test"
echo $(slurm_submit $COMMAND)