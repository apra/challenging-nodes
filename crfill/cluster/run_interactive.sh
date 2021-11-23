#!/bin/bash
JOBS_SOURCE="$HOME/challenging-nodes/crfill"
SINGULARITYIMAGE="$HOME/image_wallace.sif"
DATA="$TMPDIR/spapa"

LOGGING_DIR="$HOME/challenging-nodes/crfill/checkpoints"

singularity shell --nv \
    --bind "$DATA":/data \
    --bind $LOGGING_DIR:"$LOGGING_DIR" \
    --bind "$JOBS_SOURCE" \
    --bind "$DATA" \
    --pwd "$JOBS_SOURCE" \
    $SINGULARITYIMAGE

NAME=batchsize_64_2gpus

python -u train.py --dataset_mode_train custom_train --name $NAME --checkpoints_dir checkpoints/$NAME --dataset_mode custom_train --train_image_dir /scratch/spapa/data/images --train_nodule_list /scratch/spapa/data/metadata.csv --netG twostagend --netD deepfill --preprocess_mode none --validation_freq 100 --gpu_ids 0,1 --niter 50 --batchSize 64 --display_freq 20 --model arrange
