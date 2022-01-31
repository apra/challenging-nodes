#!/bin/bash
#SBATCH -t 2-15:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=ericj.marcus@gmail.com
#SBATCH -p gpu_titanrtx
#SBATCH --mem=150G
#SBATCH --gpus-per-node=titanrtx:4
#SBATCH --cpus-per-task=24
#SBATCH -N 1

DATE=$(date '+%Y%m%d%H%M')
EXPERIMENT_NAME="ssim831_long"

JOBS_SOURCE="$HOME/challenging-nodes/crfill"
SINGULARITYIMAGE="$HOME/crfill.sif"
DATA="/scratch/scratchdata"

#Beta for the first task
BETAL1_1=8.

#Lambda for the first task
LAMBDA_1=3.

#lambda ssim
LAMBDA_S=1.

NUM_WORKERS=5

LOGGING_DIR="$HOME/challenging-nodes/crfill/checkpoints"

#Create output directory on scratch
mkdir -p "$DATA"
cp -ra $HOME/data/. "$DATA"

NAME="$EXPERIMENT_NAME""$DATE""-beta_l$BETAL1_1-lambda_ref$LAMBDA_1-lambda_ssim$LAMBDA_S"

STANDARD_PARAMS="--seed 0 --batchSize 88 --niter_decay 70 --ssim_loss --include_chexpert --include_mimic --node21_resample_count 10 --dataset_mode_train custom_train --dataset_mode custom_train --train_image_dir /data --netG twostagend --netD deepfill --preprocess_mode none --validation_freq 20000 --niter 110 --display_freq 2000 --model arrange"

COMMAND="python -u train.py --name $NAME --num_workers $NUM_WORKERS --checkpoints_dir $LOGGING_DIR/$NAME --gpu_ids 0,1,2,3 --beta_l1 $BETAL1_1 --lambda_ref $LAMBDA_1 --lambda_ssim $LAMBDA_S $STANDARD_PARAMS"

echo "Running $NAME"
singularity exec --no-home --nv \
--bind "$DATA":/data \
--bind $LOGGING_DIR:"$LOGGING_DIR" \
--bind "$JOBS_SOURCE" \
--pwd "$JOBS_SOURCE" \
$SINGULARITYIMAGE \
$COMMAND

echo "$NAME command ran"