#!/bin/bash
#SBATCH -t 10:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=samuele.papa@gmail.com
#SBATCH -p gpu_titanrtx
#SBATCH --mem=150G
#SBATCH --gpus-per-node=titanrtx:4
#SBATCH --cpus-per-task=24
#SBATCH -N 1
#SBATCH --output=vae_%A.out
#SBATCH --error=vae_%A.err

DATE=$(date '+%Y%m%d%H%M')
EXPERIMENT_NAME="vae_4CNNs_"

JOBS_SOURCE="$HOME/challenging-nodes/crfill"
SINGULARITYIMAGE="$HOME/crfill.sif"
DATA="$TMPDIR/spapa"
TRAINING_SCRIPT="train_vae.py"

#Default 0.06
SIGMA_1=$1

#Default 1
BETAKL_1=$2

#Default 64
LATENTSIZE_1=$3

#Default 0.0005
LEARNINGRATE_1=$4

#Default 0.06
SIGMA_2=$5

#Default 1
BETAKL_2=$6

#Default 64
LATENTSIZE_2=$7

#Default 0.0005
LEARNINGRATE_2=$8

#Default 0.06
SIGMA_3=$9

#Default 1
BETAKL_3="${10}"

#Default 64
LATENTSIZE_3="${11}"

#Default 0.0005
LEARNINGRATE_3="${12}"

#Default 0.06
SIGMA_4="${13}"

#Default 1
BETAKL_4="${14}"

#Default 64
LATENTSIZE_4="${15}"

#Default 0.0005
LEARNINGRATE_4="${16}"

NUM_WORKERS=5

LOGGING_DIR="$HOME/challenging-nodes/crfill/checkpoints"

#Create output directory on scratch
mkdir -p "$DATA"
cp -ra /project/spapa/dataset_node21/ct_patches/images/. "$DATA"

NAME="$EXPERIMENT_NAME""$DATE""-beta$BETAKL_1-sigma$SIGMA_1-latentsize$LATENTSIZE_1-lr$LEARNINGRATE_1"

STANDARD_PARAMS="--seed 0 --batchSize 64 --niter_decay 100 --dataset_mode_train custom_train_vae --dataset_mode custom_train_vae --train_image_dir /data --preprocess_mode none --validation_freq 200000 --niter 900 --display_freq 10000 --model vae"

GPUID=0

COMMAND="python -u $TRAINING_SCRIPT --name $NAME --num_workers $NUM_WORKERS  --checkpoints_dir $LOGGING_DIR/$NAME --gpu_ids $GPUID --beta_kl $BETAKL_1 --lr $LEARNINGRATE_1 --sigma $SIGMA_1 --latent_size $LATENTSIZE_1 $STANDARD_PARAMS"

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

NAME="$EXPERIMENT_NAME""$DATE""-beta$BETAKL_2-sigma$SIGMA_2-latentsize$LATENTSIZE_2-lr$LEARNINGRATE_2"

GPUID=1

COMMAND="python -u $TRAINING_SCRIPT --name $NAME --num_workers $NUM_WORKERS  --checkpoints_dir $LOGGING_DIR/$NAME --gpu_ids $GPUID --beta_kl $BETAKL_2 --lr $LEARNINGRATE_2 --sigma $SIGMA_2 --latent_size $LATENTSIZE_2 $STANDARD_PARAMS"

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

NAME="$EXPERIMENT_NAME""$DATE""-beta$BETAKL_3-sigma$SIGMA_3-latentsize$LATENTSIZE_3-lr$LEARNINGRATE_3"

GPUID=2

COMMAND="python -u $TRAINING_SCRIPT --name $NAME --num_workers $NUM_WORKERS  --checkpoints_dir $LOGGING_DIR/$NAME --gpu_ids $GPUID --beta_kl $BETAKL_3 --lr $LEARNINGRATE_3 --sigma $SIGMA_3 --latent_size $LATENTSIZE_3 $STANDARD_PARAMS"

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

NAME="$EXPERIMENT_NAME""$DATE""-beta$BETAKL_4-sigma$SIGMA_4-latentsize$LATENTSIZE_4-lr$LEARNINGRATE_4"

GPUID=3

COMMAND="python -u $TRAINING_SCRIPT --name $NAME --num_workers $NUM_WORKERS  --checkpoints_dir $LOGGING_DIR/$NAME --gpu_ids $GPUID --beta_kl $BETAKL_4 --lr $LEARNINGRATE_4 --sigma $SIGMA_4 --latent_size $LATENTSIZE_4 $STANDARD_PARAMS"

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