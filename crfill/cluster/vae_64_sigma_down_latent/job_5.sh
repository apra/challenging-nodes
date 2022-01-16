#!/bin/bash
#SBATCH -t 10:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=samuele.papa@gmail.com
#SBATCH -p gpu_titanrtx
#SBATCH --mem=75G
#SBATCH --gpus-per-node=titanrtx:4
#SBATCH --cpus-per-task=12
#SBATCH -N 1
#SBATCH --output=vae_64_sigma_down_latent_5%A.out
#SBATCH --error=vae_64_sigma_down_latent_5%A.err

DATA="$TMPDIR/spapa"

#Create output directory on scratch
mkdir -p "$DATA"
cp -ra /project/spapa/dataset_node21/. "$DATA"
singularity exec --no-home --nv \
--bind $TMPDIR/spapa:/data  \
--bind $HOME/challenging-nodes/crfill  \
--bind $HOME/challenging-nodes/crfill/checkpoints  \
--pwd $HOME/challenging-nodes/crfill  \
$HOME/crfill.sif  \
python -u train_vae.py --dataset_mode_train custom_train_vae --dataset_mode custom_train_vae --train_image_dir /data/images --preprocess_mode none --validation_freq 50000 --display_freq 10000 --model vae --batchSize 64\
 --niter_decay 500\
 --niter 1000\
 --sigma 0.01\
 --beta_kl 0.0012980119315177806\
 --latent_size 32\
 --downsample 4\
 --seed 0\
 --num_workers 5\
 --lr 0.0001\
 --name vae_64_sigma_down_latent_20 --checkpoints_dir $HOME/challenging-nodes/crfill/checkpoints/vae_64_sigma_down_latent_20 --gpu_ids 0 &
singularity exec --no-home --nv \
--bind $TMPDIR/spapa:/data  \
--bind $HOME/challenging-nodes/crfill  \
--bind $HOME/challenging-nodes/crfill/checkpoints  \
--pwd $HOME/challenging-nodes/crfill  \
$HOME/crfill.sif  \
python -u train_vae.py --dataset_mode_train custom_train_vae --dataset_mode custom_train_vae --train_image_dir /data/images --preprocess_mode none --validation_freq 50000 --display_freq 10000 --model vae --batchSize 64\
 --niter_decay 500\
 --niter 1000\
 --sigma 0.01\
 --beta_kl 0.0031414713692677143\
 --latent_size 8\
 --downsample 4\
 --seed 0\
 --num_workers 5\
 --lr 0.0001\
 --name vae_64_sigma_down_latent_21 --checkpoints_dir $HOME/challenging-nodes/crfill/checkpoints/vae_64_sigma_down_latent_21 --gpu_ids 1 &
singularity exec --no-home --nv \
--bind $TMPDIR/spapa:/data  \
--bind $HOME/challenging-nodes/crfill  \
--bind $HOME/challenging-nodes/crfill/checkpoints  \
--pwd $HOME/challenging-nodes/crfill  \
$HOME/crfill.sif  \
python -u train_vae.py --dataset_mode_train custom_train_vae --dataset_mode custom_train_vae --train_image_dir /data/images --preprocess_mode none --validation_freq 50000 --display_freq 10000 --model vae --batchSize 64\
 --niter_decay 500\
 --niter 1000\
 --sigma 0.01\
 --beta_kl 0.4813906588782617\
 --latent_size 64\
 --downsample 4\
 --seed 0\
 --num_workers 5\
 --lr 0.0001\
 --name vae_64_sigma_down_latent_22 --checkpoints_dir $HOME/challenging-nodes/crfill/checkpoints/vae_64_sigma_down_latent_22 --gpu_ids 2 &
singularity exec --no-home --nv \
--bind $TMPDIR/spapa:/data  \
--bind $HOME/challenging-nodes/crfill  \
--bind $HOME/challenging-nodes/crfill/checkpoints  \
--pwd $HOME/challenging-nodes/crfill  \
$HOME/crfill.sif  \
python -u train_vae.py --dataset_mode_train custom_train_vae --dataset_mode custom_train_vae --train_image_dir /data/images --preprocess_mode none --validation_freq 50000 --display_freq 10000 --model vae --batchSize 64\
 --niter_decay 500\
 --niter 1000\
 --sigma 0.01\
 --beta_kl 0.3879341802770623\
 --latent_size 8\
 --downsample 2\
 --seed 0\
 --num_workers 5\
 --lr 0.0001\
 --name vae_64_sigma_down_latent_23 --checkpoints_dir $HOME/challenging-nodes/crfill/checkpoints/vae_64_sigma_down_latent_23 --gpu_ids 3