#!/bin/bash
JOBS_SOURCE="$HOME/challenging-nodes/crfill"
SINGULARITYIMAGE="$HOME/crfill.sif"
DATA="$TMPDIR/spapa"

mkdir -p "$DATA"
cp -r $HOME/data/ "$DATA"

LOGGING_DIR="$HOME/challenging-nodes/crfill/checkpoints"

singularity shell --no-home --nv \
    --bind "$DATA":/data \
    --bind $LOGGING_DIR:"$LOGGING_DIR" \
    --bind "$JOBS_SOURCE" \
    --bind "$DATA" \
    --pwd "$JOBS_SOURCE" \
    $SINGULARITYIMAGE

NAME=batchsize_48_2gpus

STANDARD_PARAMS="--dataset_mode_train custom_train --dataset_mode custom_train --train_image_dir /scratch/spapa/data/images --train_nodule_list /scratch/spapa/data/metadata.csv --netG twostagend --netD deepfill --preprocess_mode none --validation_freq 2000 --niter 100 --display_freq 500 --model arrange"

python -u train.py --name batchsize_48_3gpus --checkpoints_dir checkpoints/batchsize_48_3gpus --gpu_ids 0,1 --batchSize 96 $STANDARD_PARAMS