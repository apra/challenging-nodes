#!/bin/bash
#SBATCH -t 10:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=samuele.papa@gmail.com
#SBATCH -p gpu_titanrtx
#SBATCH --mem=150G
#SBATCH --gpus-per-node=titanrtx:4
#SBATCH --cpus-per-task=24
#SBATCH -N 1
#SBATCH --output=vae_sigma_beta_param_search_3%A.out
#SBATCH --error=vae_sigma_beta_param_search_3%A.err

DATA="$TMPDIR/spapa"

#Create output directory on scratch
mkdir -p "$DATA"
cp -ra /project/spapa/dataset_node21/ct_patches/images/. "$DATA"
singularity exec --no-home --nv --bind $TMPDIR/spapa:/data --bind $HOME/challenging-nodes/crfill --bind $HOME/challenging-nodes/crfill/checkpoints --pwd $HOME/challenging-nodes/crfill $HOME/crfill.sif python -u train_vae.py --dataset_mode_train custom_train_vae --dataset_mode custom_train_vae --train_image_dir /data --preprocess_mode none --validation_freq 50000 --display_freq 10000 --model vae --batchSize 64 --niter_decay 500 --niter 1000 --sigma 2.6891628709146724 --beta_kl 0.5522909641366669 --latent_size 64 --downsample 2 --seed 0 --num_workers 5 --lr 0.0001 --name vae_sigma_beta_param_search_12 --checkpoints_dir $HOME/challenging-nodes/crfill/checkpoints/vae_sigma_beta_param_search_12 --gpu_ids 0 &
singularity exec --no-home --nv --bind $TMPDIR/spapa:/data --bind $HOME/challenging-nodes/crfill --bind $HOME/challenging-nodes/crfill/checkpoints --pwd $HOME/challenging-nodes/crfill $HOME/crfill.sif python -u train_vae.py --dataset_mode_train custom_train_vae --dataset_mode custom_train_vae --train_image_dir /data --preprocess_mode none --validation_freq 50000 --display_freq 10000 --model vae --batchSize 64 --niter_decay 500 --niter 1000 --sigma 0.0013625181657841055 --beta_kl 0.3997947668399872 --latent_size 64 --downsample 2 --seed 0 --num_workers 5 --lr 0.0001 --name vae_sigma_beta_param_search_13 --checkpoints_dir $HOME/challenging-nodes/crfill/checkpoints/vae_sigma_beta_param_search_13 --gpu_ids 1 &
singularity exec --no-home --nv --bind $TMPDIR/spapa:/data --bind $HOME/challenging-nodes/crfill --bind $HOME/challenging-nodes/crfill/checkpoints --pwd $HOME/challenging-nodes/crfill $HOME/crfill.sif python -u train_vae.py --dataset_mode_train custom_train_vae --dataset_mode custom_train_vae --train_image_dir /data --preprocess_mode none --validation_freq 50000 --display_freq 10000 --model vae --batchSize 64 --niter_decay 500 --niter 1000 --sigma 0.8291283895033408 --beta_kl 0.567266219538856 --latent_size 64 --downsample 2 --seed 0 --num_workers 5 --lr 0.0001 --name vae_sigma_beta_param_search_14 --checkpoints_dir $HOME/challenging-nodes/crfill/checkpoints/vae_sigma_beta_param_search_14 --gpu_ids 2 &
singularity exec --no-home --nv --bind $TMPDIR/spapa:/data --bind $HOME/challenging-nodes/crfill --bind $HOME/challenging-nodes/crfill/checkpoints --pwd $HOME/challenging-nodes/crfill $HOME/crfill.sif python -u train_vae.py --dataset_mode_train custom_train_vae --dataset_mode custom_train_vae --train_image_dir /data --preprocess_mode none --validation_freq 50000 --display_freq 10000 --model vae --batchSize 64 --niter_decay 500 --niter 1000 --sigma 0.005042228030788948 --beta_kl 0.035948907446999856 --latent_size 64 --downsample 2 --seed 0 --num_workers 5 --lr 0.0001 --name vae_sigma_beta_param_search_15 --checkpoints_dir $HOME/challenging-nodes/crfill/checkpoints/vae_sigma_beta_param_search_15 --gpu_ids 3