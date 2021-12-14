#!/bin/bash
export CUDA_VISIBLE_DEVICES=4,5  # specify gpu
export OMP_NUM_THREADS=20  # threads

NUM_WORKERS=5
NAME=rhpc_ssim551_rcnn_finetune

LOGGING_DIR="$HOME/projects/crfill/checkpoints"
GENERATOR_LOC="./checkpoints/$NAME/$NAME/latest_net_G.pth"
DISCRIMINATOR_LOC="./checkpoints/$NAME/$NAME/latest_net_D.pth"

STANDARD_PARAMS="--seed 0  --batchSize 40 --load_base_g $GENERATOR_LOC --load_base_d $DISCRIMINATOR_LOC --ssim_loss --niter_decay 15 --include_chexpert --include_mimic --node21_resample_count 10 --dataset_mode_train custom_train_doubledisc --dataset_mode custom_train --train_image_dir /data --netG twostagend --netD deepfill --preprocess_mode none --validation_freq 20000 --niter 30 --display_freq 2000 --model arrangedoubledisc"
#STANDARD_PARAMS="--seed 0  --batchSize 40  --dataset_mode_train custom_train_negative --dataset_mode custom_train_negative --train_image_dir /data --netG twostagend --netD deepfill --preprocess_mode none --validation_freq 20000 --niter 600 --display_freq 2000 --model arrange"

python -u train_doubledisc.py --name $NAME --num_workers $NUM_WORKERS --checkpoints_dir $LOGGING_DIR/$NAME --gpu_ids 0,1 --beta_l1 5. --lambda_ref 5. --lambda_ssim 1. --lambda_rcnn 1. $STANDARD_PARAMS
