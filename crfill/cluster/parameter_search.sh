#!/bin/bash
#SBATCH -t 11:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=samuele.papa@gmail.com
#SBATCH -p gpu_titanrtx
#SBATCH --mem=150G
#SBATCH --gpus-per-node=titanrtx:4
#SBATCH --cpus-per-task=24
#SBATCH -N 1
#SBATCH --output=run_%A.out
#SBATCH --error=run_%A.err

JOBS_SOURCE="$HOME/challenging-nodes/crfill"
SINGULARITYIMAGE="$HOME/crfill.sif"
DATA="$TMPDIR/spapa"

#Beta for the first task
BETAL1_1=$1
#Beta for the second task
BETAL1_2=$2

#Lambda for the first task
LAMBDA_1=$3
#Lambda for the second task
LAMBDA_2=$4

NUM_WORKERS=5

LOGGING_DIR="$HOME/challenging-nodes/crfill/checkpoints"

#Create output directory on scratch
mkdir -p "$DATA"
cp -ra $HOME/data/data_node21/. "$DATA"

NAME="test_parametersearch_full_dataset_2gpus_beta_l$BETAL1_1-lambda_feat$LAMBDA_1"

STANDARD_PARAMS="--seed 0 --batchSize 40 --niter_decay 60 --include_chexpert --include_mimic --node21_resample_count 10 --dataset_mode_train custom_train --dataset_mode custom_train --train_image_dir /data --netG twostagend --netD deepfill --preprocess_mode none --validation_freq 20000 --niter 600 --display_freq 2000 --model arrange"

COMMAND="python -u train.py --name $NAME --num_workers $NUM_WORKERS --checkpoints_dir $LOGGING_DIR/$NAME --gpu_ids 0,1 --beta_l1 $BETAL1_1 --lambda_feat $LAMBDA_1 $STANDARD_PARAMS"

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
NAME="test_parametersearch_full_dataset_2gpus_beta_l$BETAL1_2-lambda_feat$LAMBDA_2"

COMMAND="python -u train.py --name $NAME --num_workers $NUM_WORKERS --checkpoints_dir $LOGGING_DIR/$NAME --gpu_ids 2,3 --beta_l1 $BETAL1_2 --lambda_feat $LAMBDA_2 $STANDARD_PARAMS"

echo "Running $NAME"
singularity exec --no-home --nv \
--bind "$DATA":/data \
--bind $LOGGING_DIR:"$LOGGING_DIR" \
--bind "$JOBS_SOURCE" \
--bind "$DATA" \
--pwd "$JOBS_SOURCE" \
$SINGULARITYIMAGE \
$COMMAND

echo "$NAME command ran"