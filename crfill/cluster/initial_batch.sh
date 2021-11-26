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
NODE21="$DATA/node21"

LOGGING_DIR="$HOME/challenging-nodes/crfill/checkpoints"

#Create output directory on scratch
mkdir -p "$DATA"
mkdir -p "$NODE21"
cp -ra $HOME/data/. "$NODE21"

NAME=batchsize_48_2gpus_beta_l15_lambda_feat05

STANDARD_PARAMS="--dataset_mode_train custom_train --dataset_mode custom_train --train_image_dir /data --netG twostagend --netD deepfill --preprocess_mode none --validation_freq 2000 --niter 100 --display_freq 500 --model arrange"

COMMAND="python -u train.py --name $NAME --num_workers 4 --checkpoints_dir $LOGGING_DIR/$NAME --gpu_ids 0,1 --beta_l1 1.5 --lambda_feat 0.5 --batchSize 48 $STANDARD_PARAMS"

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
NAME=batchsize_48_2gpus_beta_l15

STANDARD_PARAMS="--dataset_mode_train custom_train --dataset_mode custom_train --train_image_dir /data --netG twostagend --netD deepfill --preprocess_mode none --validation_freq 1000 --niter 600 --display_freq 1000 --model arrange"

COMMAND="python -u train.py --name $NAME --num_workers 4 --checkpoints_dir $LOGGING_DIR/$NAME --gpu_ids 2,3 --beta_l1 1.5 --batchSize 48 $STANDARD_PARAMS"

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