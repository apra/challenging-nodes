#!/bin/bash
export CUDA_VISIBLE_DEVICES=4,5  # specify gpu
export OMP_NUM_THREADS=20  # threads

NUM_WORKERS=5

LOGGING_DIR="$HOME/projects/crfill/checkpoints"

NAME=rhpc_test_run_skipconn

#Beta for the first task
BETAL1_1=5.

#Lambda for the first task
LAMBDA_1=5.

#Lambda rcnn
LAMBDARCNN_1=1.


STANDARD_PARAMS="--seed 0 --batchSize 30 --gan_mode hinge --niter_decay 15 --include_chexpert --include_mimic --node21_resample_count 10 --dataset_mode_train custom_train_all --dataset_mode custom_train --train_image_dir /data --netG twostagend --netD deepfill --preprocess_mode none --validation_freq 20000 --niter 43 --display_freq 2000 --model arrangeskipconn"

python -u train_doubledisc.py --name $NAME --num_workers $NUM_WORKERS --checkpoints_dir $LOGGING_DIR/$NAME --gpu_ids 0,1 --beta_l1 $BETAL1_1 --lambda_ref $LAMBDA_1 --lambda_rcnn $LAMBDARCNN_1 $STANDARD_PARAMS
