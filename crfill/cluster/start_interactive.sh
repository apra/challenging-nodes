#!/bin/bash

NAME=batchsize_48_2gpus

JOBS_SOURCE="$HOME/challenging-nodes/crfill"
SINGULARITYIMAGE="$HOME/crfill.sif"
DATA="$TMPDIR/spapa"

LOGGING_DIR="$HOME/challenging-nodes/crfill/checkpoints"

STANDARD_PARAMS="--dataset_mode_train custom_train --dataset_mode custom_train --train_image_dir /data/images --train_nodule_list /data/metadata.csv --netG twostagend --netD deepfill --preprocess_mode none --validation_freq 2000 --niter 100 --display_freq 500 --model arrange"

python -u train.py --name $NAME --num_workers 16 --checkpoints_dir $LOGGING_DIR/$NAME --gpu_ids 0,1 --batchSize 48 $STANDARD_PARAMS
