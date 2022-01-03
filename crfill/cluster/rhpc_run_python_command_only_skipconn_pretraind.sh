#!/bin/bash
export CUDA_VISIBLE_DEVICES=4,5  # specify gpu
export OMP_NUM_THREADS=20  # threads

NUM_WORKERS=5

LOGGING_DIR="$HOME/projects/crfill/checkpoints"

NAME=rhpc_test_run_skipconn_pretraind

#Beta for the first task
BETAL1_1=0.

#Lambda for the first task
LAMBDA_1=0.

#lambda ssim
LAMBDA_S=0.

#Lambda rcnn
LAMBDARCNN_1=0.

DISCRIMINATOR_LOC="$SCRATCH/chalnode_checkpoints/disc_masked_aux_not/latest_net_D.pth"

GENERATOR_LOC="$SCRATCH/chalnode_checkpoints/disc_masked_aux_not/latest_net_G.pth"


#STANDARD_PARAMS="--seed 0 --batchSize 40 --gan_mode hinge --mask_pos_discriminator --niter_decay 15 --load_base_d $DISCRIMINATOR_LOC --load_base_g $GENERATOR_LOC --ssim_loss --include_chexpert --include_mimic --node21_resample_count 10 --dataset_mode_train custom_train_all_pos_masked --dataset_mode custom_train --train_image_dir /data --netG twostagend --netD deepfill --preprocess_mode none --validation_freq 20000 --niter 120 --display_freq 2000 --model arrangeskipconn"
STANDARD_PARAMS="--seed 0 --batchSize 30 --gan_mode debug --mask_pos_discriminator --niter_decay 15 --load_base_g $GENERATOR_LOC --load_base_d $DISCRIMINATOR_LOC --ssim_loss --dataset_mode_train custom_train_all_pos_masked --dataset_mode custom_train --train_image_dir /data --netG twostagend --netD deepfill --preprocess_mode none --validation_freq 20000 --niter 120 --display_freq 100 --model arrangeskipconn"

python -u train_doubledisc.py --name $NAME --num_workers $NUM_WORKERS --checkpoints_dir $LOGGING_DIR/$NAME --gpu_ids 0,1 --beta_l1 $BETAL1_1 --lambda_ref $LAMBDA_1 --lambda_rcnn $LAMBDARCNN_1 --lambda_ssim $LAMBDA_S $STANDARD_PARAMS
