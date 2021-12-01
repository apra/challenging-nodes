#!/bin/bash
export CUDA_VISIBLE_DEVICES=4  # specify gpu
export OMP_NUM_THREADS=20  # threads

JOBS_SOURCE="$HOME/projects/crfill"
SINGULARITYIMAGE="$HOME/projects/singularity_files/crfill.sif"
DATA="$SCRATCH"

NUM_WORKERS=5

LOGGING_DIR="$HOME/projects/crfill/checkpoints"

NAME=rhpc_test_run

STANDARD_PARAMS="--seed 0  --batchSize 40 --include_chexpert --include_mimic --node21_resample_count 10 --dataset_mode_train custom_train --dataset_mode custom_train --train_image_dir /data --netG twostagend --netD deepfill --preprocess_mode none --validation_freq 20000 --niter 600 --display_freq 2000 --model arrange"

COMMAND="python -u train.py --name $NAME --num_workers $NUM_WORKERS --checkpoints_dir $LOGGING_DIR/$NAME --gpu_ids 0 --beta_l1 1.5 --lambda_feat 0.5 $STANDARD_PARAMS"

echo "Running $NAME"
singularity exec --no-home --nv \
--bind "$DATA":/data \
--bind $LOGGING_DIR:"$LOGGING_DIR" \
--bind "$JOBS_SOURCE" \
--bind "$DATA" \
--pwd "$JOBS_SOURCE" \
$SINGULARITYIMAGE \
$COMMAND &
echo "$NAME command ran"