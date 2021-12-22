#!/bin/bash
#SBATCH -t 10:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=samuele.papa@gmail.com
#SBATCH -p gpu_titanrtx
#SBATCH --mem=150G
#SBATCH --gpus-per-node=titanrtx:4
#SBATCH --cpus-per-task=24
#SBATCH -N 1
#SBATCH --output=vae_sigma_beta_param_search_0%A.out
#SBATCH --error=vae_sigma_beta_param_search_0%A.err

DATA="$TMPDIR/spapa"

#Create output directory on scratch
mkdir -p "$DATA"
cp -ra /project/spapa/dataset_node21/ct_patches/images/. "$DATA"
singularity exec --no-home --nv --bing $TMPDIR/spapa:/data --bind $HOME/challenging-nodes/crfill --bind $HOME/challenging-nodes/crfill/checkpoints/vae_sigma_beta_param_search_0 --pwd $HOME/challenging-nodes/crfill $HOME/crfill.sif python -u train_vae.py --dataset_mode_train custom_train_vae --dataset_mode custom_train_vae --train_image_dir /data --preprocess_mode none --validation_freq 50000 --display_freq 10000 --model vae --batchSize 64 --niter_decay 500 --niter 1000 --sigma 0.3530585630408593 --beta_kl 2.836061794577942 --latent_size 64 --downsample 2 --seed 0 --num_workers 5 --lr 0.0001 --name vae_sigma_beta_param_search_0 --checkpoints_dir $HOME/challenging-nodes/crfill/checkpoints/vae_sigma_beta_param_search_0 --gpu_ids 0
singularity exec --no-home --nv --bing $TMPDIR/spapa:/data --bind $HOME/challenging-nodes/crfill --bind $HOME/challenging-nodes/crfill/checkpoints/vae_sigma_beta_param_search_1 --pwd $HOME/challenging-nodes/crfill $HOME/crfill.sif python -u train_vae.py --dataset_mode_train custom_train_vae --dataset_mode custom_train_vae --train_image_dir /data --preprocess_mode none --validation_freq 50000 --display_freq 10000 --model vae --batchSize 64 --niter_decay 500 --niter 1000 --sigma 0.011999049779393502 --beta_kl 0.14650244778091628 --latent_size 64 --downsample 2 --seed 0 --num_workers 5 --lr 0.0001 --name vae_sigma_beta_param_search_1 --checkpoints_dir $HOME/challenging-nodes/crfill/checkpoints/vae_sigma_beta_param_search_1 --gpu_ids 1
singularity exec --no-home --nv --bing $TMPDIR/spapa:/data --bind $HOME/challenging-nodes/crfill --bind $HOME/challenging-nodes/crfill/checkpoints/vae_sigma_beta_param_search_2 --pwd $HOME/challenging-nodes/crfill $HOME/crfill.sif python -u train_vae.py --dataset_mode_train custom_train_vae --dataset_mode custom_train_vae --train_image_dir /data --preprocess_mode none --validation_freq 50000 --display_freq 10000 --model vae --batchSize 64 --niter_decay 500 --niter 1000 --sigma 0.001458458566595887 --beta_kl 0.01580693116829211 --latent_size 64 --downsample 2 --seed 0 --num_workers 5 --lr 0.0001 --name vae_sigma_beta_param_search_2 --checkpoints_dir $HOME/challenging-nodes/crfill/checkpoints/vae_sigma_beta_param_search_2 --gpu_ids 2
singularity exec --no-home --nv --bing $TMPDIR/spapa:/data --bind $HOME/challenging-nodes/crfill --bind $HOME/challenging-nodes/crfill/checkpoints/vae_sigma_beta_param_search_3 --pwd $HOME/challenging-nodes/crfill $HOME/crfill.sif python -u train_vae.py --dataset_mode_train custom_train_vae --dataset_mode custom_train_vae --train_image_dir /data --preprocess_mode none --validation_freq 50000 --display_freq 10000 --model vae --batchSize 64 --niter_decay 500 --niter 1000 --sigma 0.0011644223751767875 --beta_kl 0.04906241083254889 --latent_size 64 --downsample 2 --seed 0 --num_workers 5 --lr 0.0001 --name vae_sigma_beta_param_search_3 --checkpoints_dir $HOME/challenging-nodes/crfill/checkpoints/vae_sigma_beta_param_search_3 --gpu_ids 3