#!/bin/bash
export CUDA_VISIBLE_DEVICES=2,3  # specify gpu
export OMP_NUM_THREADS=20  # threads

NUM_WORKERS=5

LOGGING_DIR="/proj/checkpoints"

NAME=rhpc_ssim555_new_normalization

STANDARD_PARAMS="--seed 0  --batchSize 80 --ssim_loss --niter_decay 15 --include_chexpert --include_mimic --node21_resample_count 10 --dataset_mode_train custom_train --dataset_mode custom_train --train_image_dir /mnt --netG twostagend --netD deepfill --preprocess_mode none --validation_freq 20000 --niter 50 --display_freq 2000 --model arrange"
#STANDARD_PARAMS="--seed 0  --batchSize 40  --dataset_mode_train custom_train_negative --dataset_mode custom_train_negative --train_image_dir /data --netG twostagend --netD deepfill --preprocess_mode none --validation_freq 20000 --niter 600 --display_freq 2000 --model arrange"

python -u train.py --name $NAME --num_workers $NUM_WORKERS --checkpoints_dir $LOGGING_DIR/$NAME --gpu_ids 0,1 --beta_l1 5. --lambda_ref 5. --lambda_ssim 1. $STANDARD_PARAMS
