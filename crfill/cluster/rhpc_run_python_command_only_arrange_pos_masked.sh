#!/bin/bash
export CUDA_VISIBLE_DEVICES=4,5  # specify gpu
export OMP_NUM_THREADS=20  # threads

NUM_WORKERS=5

LOGGING_DIR="$HOME/projects/crfill/checkpoints"

NAME=rhpc_disc_masked_aux_not

#Beta for the first task
BETAL1_1=7.

#Lambda for the first task
LAMBDA_1=3.

#lambda ssim
LAMBDA_S=0.9

#Lambda rcnn
LAMBDARCNN_1=0.


STANDARD_PARAMS="--seed 0 --batchSize 80 --gan_mode hinge --mask_pos_discriminator --niter_decay 20 --ssim_loss --include_chexpert --include_mimic --node21_resample_count 10 --dataset_mode_train custom_train_pos_masked --dataset_mode custom_train --train_image_dir /data --netG twostagend --netD deepfill --preprocess_mode none --validation_freq 20000 --niter 60 --display_freq 2000 --model arrange"


python -u train.py --name $NAME --num_workers $NUM_WORKERS --checkpoints_dir $LOGGING_DIR/$NAME --gpu_ids 0,1 --beta_l1 $BETAL1_1 --lambda_ref $LAMBDA_1 --lambda_rcnn $LAMBDARCNN_1 --lambda_ssim $LAMBDA_S $STANDARD_PARAMS
