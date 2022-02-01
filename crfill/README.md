# crfill

The original paper can be found [here](https://arxiv.org/pdf/2011.12836.pdf) | Below is an example script that continues training:
```shell
#!/bin/bash
#SBATCH -t 2-00:00:00
#SBATCH --mem=75G
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --output=train_with_flip_%A.out
#SBATCH --error=train_with_flip_%A.err

NUM_WORKERS=5
SINGULARITYIMAGE="singularity_file.sif"
LOGGING_DIR="/logdir"
JOBS_SOURCE="/projectdir"

NAME=example_continue_training

DATA_LOC="/data/node21_data"
DISCRIMINATOR_LOC="latest_net_D.pth"
GENERATOR_LOC="latest_net_G.pth"

STANDARD_PARAMS="--seed 0  --batchSize 80 --ssim_loss --lr 0.00009 --niter_decay 300 --load_base_g $GENERATOR_LOC --load_base_d $DISCRIMINATOR_LOC --include_chexpert --include_mimic --node21_resample_count 10 --dataset_mode_train custom_train --dataset_mode custom_train --train_image_dir $DATA_LOC --netG twostagend --netD deepfill --preprocess_mode none --validation_freq 20000 --niter 2 --display_freq 2000 --model arrange"

COMMAND="python -u train.py --name $NAME --num_workers $NUM_WORKERS --checkpoints_dir $LOGGING_DIR/$NAME --gpu_ids 0,1 --beta_l1 5. --lambda_ref 5. --lambda_ssim 1. $STANDARD_PARAMS"

echo "Running $NAME"
singularity exec --no-home --nv \
--bind "$JOBS_SOURCE":/proj \
--pwd /proj \
$SINGULARITYIMAGE \
$COMMAND

echo "$NAME command ran"
```
