#!/bin/bash
export CUDA_VISIBLE_DEVICES=4,6  # specify gpu
export OMP_NUM_THREADS=5  # threads

NUM_WORKERS=5

LOGGING_DIR="$HOME/projects/crfill/checkpoints"

NAME=rhpc_test_run_sem_creator

STANDARD_PARAMS="--seed 0  --batchSize 90 --k_means_clusters 3 --include_chexpert --include_mimic --node21_resample_count 10 --dataset_mode_train custom_train_semantic_creator --dataset_mode custom_train --train_image_dir /data --netG twostagend --netD deepfill --preprocess_mode none --validation_freq 20000 --niter 600 --display_freq 2000 --model arrange"
#STANDARD_PARAMS="--seed 0  --batchSize 40  --dataset_mode_train custom_train_negative --dataset_mode custom_train_negative --train_image_dir /data --netG twostagend --netD deepfill --preprocess_mode none --validation_freq 20000 --niter 600 --display_freq 2000 --model arrange"

python -u train.py --name $NAME --num_workers $NUM_WORKERS --checkpoints_dir $LOGGING_DIR/$NAME --gpu_ids 0,1 --beta_l1 10. --lambda_ref 2. $STANDARD_PARAMS
