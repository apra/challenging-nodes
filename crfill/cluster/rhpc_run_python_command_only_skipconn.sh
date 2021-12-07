#!/bin/bash
export CUDA_VISIBLE_DEVICES=4  # specify gpu
export OMP_NUM_THREADS=20  # threads

NUM_WORKERS=5

LOGGING_DIR="$HOME/projects/crfill/checkpoints"

NAME=rhpc_test_run_skipconn

STANDARD_PARAMS="--seed 0 --batchSize 10 --niter_decay 8 --include_chexpert --include_mimic --node21_resample_count 10 --dataset_mode_train custom_train_all --dataset_mode custom_train --train_image_dir /data --netG twostagend --netD deepfill --preprocess_mode none --validation_freq 20000 --niter 43 --display_freq 5000 --model arrangeskipconn"

python -u train.py --name $NAME --num_workers $NUM_WORKERS --checkpoints_dir $LOGGING_DIR/$NAME --gpu_ids 0 --beta_l1 1. --lambda_ref 1. $STANDARD_PARAMS
