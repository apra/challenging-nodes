#!/bin/bash
export CUDA_VISIBLE_DEVICES=4  # specify gpu
export OMP_NUM_THREADS=20  # threads

NUM_WORKERS=5

LOGGING_DIR="$HOME/projects/crfill/checkpoints"

NAME=rhpc_test_run

STANDARD_PARAMS="--seed 0  --batchSize 25 --dataset_mode_train custom_train_all --dataset_mode custom_train --train_image_dir /data --netG twostagend --netD deepfill --preprocess_mode none --validation_freq 20000 --niter 600 --display_freq 500 --model arrangedoubledisc"

python -u train_doubledisc.py --name $NAME --num_workers $NUM_WORKERS --checkpoints_dir $LOGGING_DIR/$NAME --gpu_ids 0 --beta_l1 1. --lambda_ref 1. --lambda_rcnn 1. $STANDARD_PARAMS
