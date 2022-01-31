#!/bin/bash
#SBATCH -t 2-00:00:00
#SBATCH --mem=75G
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --output=train_with_flip_%A.out
#SBATCH --error=train_with_flip_%A.err

NUM_WORKERS=5
SINGULARITYIMAGE="$HOME/projects/singularity_files/crfill.sif"
LOGGING_DIR="/proj/checkpoints"
JOBS_SOURCE="$HOME/projects/crfill"

NAME=continue_train_ssim_551

DATA_LOC="/processing/e.marcus/node21_data"
DISCRIMINATOR_LOC="$SCRATCH/chalnode_checkpoints/continued_training_ssim551/latest_net_D.pth"
GENERATOR_LOC="$SCRATCH/chalnode_checkpoints/continued_training_ssim551/latest_net_G.pth"

STANDARD_PARAMS="--seed 0  --batchSize 80 --ssim_loss --lr 0.00009 --niter_decay 300 --load_base_g $GENERATOR_LOC --load_base_d $DISCRIMINATOR_LOC --include_chexpert --include_mimic --node21_resample_count 10 --dataset_mode_train custom_train --dataset_mode custom_train --train_image_dir $DATA_LOC --netG twostagend --netD deepfill --preprocess_mode none --validation_freq 20000 --niter 2 --display_freq 2000 --model arrange"
#STANDARD_PARAMS="--seed 0  --batchSize 40  --dataset_mode_train custom_train_negative --dataset_mode custom_train_negative --train_image_dir /data --netG twostagend --netD deepfill --preprocess_mode none --validation_freq 20000 --niter 600 --display_freq 2000 --model arrange"

COMMAND="python -u train.py --name $NAME --num_workers $NUM_WORKERS --checkpoints_dir $LOGGING_DIR/$NAME --gpu_ids 0,1 --beta_l1 5. --lambda_ref 5. --lambda_ssim 1. $STANDARD_PARAMS"

echo "Running $NAME"
singularity exec --no-home --nv \
--bind "$JOBS_SOURCE":/proj \
--bind /processing/:/processing \
--pwd /proj \
$SINGULARITYIMAGE \
$COMMAND

echo "$NAME command ran"