#!/bin/bash
#SBATCH -t 10:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=samuele.papa@gmail.com
#SBATCH -p gpu_titanrtx
#SBATCH --mem=150G
#SBATCH --gpus-per-node=titanrtx:4
#SBATCH --cpus-per-task=24
#SBATCH -N 1
#SBATCH --output=place_lesion_%A.out
#SBATCH --error=place_lesion_%A.err

JOBS_SOURCE="$HOME/challenging-nodes/crfill"
SINGULARITYIMAGE="$HOME/crfill.sif"
DATA="$TMPDIR/spapa"
TRAINING_SCRIPT="train_placelesion.py"

NUM_WORKERS=5

LOGGING_DIR="$HOME/challenging-nodes/crfill/checkpoints"

#Create output directory on scratch
mkdir -p "$DATA"
cp -ra $HOME/data/. "$DATA"

NAME="vae_sigma_beta_param_search_4"

STANDARD_PARAMS="--no_l1_loss --network_path $LOGGING_DIR/vae_sigma_beta_param_search_4/vae_sigma_beta_param_search_4/latest_net_G.pth --seed 0 --lr 0.0002 --batchSize 32 --niter_decay 8 --include_chexpert --include_mimic --node21_resample_count 10 --dataset_mode_train custom_train_all --dataset_mode custom_train --train_image_dir /data --D_steps_per_G 100000 --netD deepfill --preprocess_mode none --validation_freq 20000 --niter 43 --display_freq 5000 --model arrangeplacelesion"

COMMAND="python -u $TRAINING_SCRIPT --load_from_checkpoint --num_folds 20 --name $NAME --num_workers $NUM_WORKERS --checkpoints_dir $LOGGING_DIR/$NAME --gpu_ids 0,1,2,3 $STANDARD_PARAMS"

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