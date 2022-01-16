#!/bin/bash
#SBATCH -t 10:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=samuele.papa@gmail.com
#SBATCH -p gpu_titanrtx
#SBATCH --mem=75G
#SBATCH --gpus-per-node=titanrtx:4
#SBATCH --cpus-per-task=12
#SBATCH -N 1
#SBATCH --output=vae_64_sigma_down_latent_2%A.out
#SBATCH --error=vae_64_sigma_down_latent_2%A.err

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
 --beta_kl 0.14945139248881537\
 --latent_size 64\
 --downsample 2\
 --seed 0\
 --num_workers 5\
 --lr 0.0001\
 --name vae_64_sigma_down_latent_8 --checkpoints_dir $HOME/challenging-nodes/crfill/checkpoints/vae_64_sigma_down_latent_8 --gpu_ids 0 &
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
 --beta_kl 5.499075661406077\
 --latent_size 64\
 --downsample 2\
 --seed 0\
 --num_workers 5\
 --lr 0.0001\
 --name vae_64_sigma_down_latent_9 --checkpoints_dir $HOME/challenging-nodes/crfill/checkpoints/vae_64_sigma_down_latent_9 --gpu_ids 1 &
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
 --beta_kl 1.8340628601973463\
 --latent_size 8\
 --downsample 4\
 --seed 0\
 --num_workers 5\
 --lr 0.0001\
 --name vae_64_sigma_down_latent_10 --checkpoints_dir $HOME/challenging-nodes/crfill/checkpoints/vae_64_sigma_down_latent_10 --gpu_ids 2 &
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
 --beta_kl 0.0010255432976732008\
 --latent_size 8\
 --downsample 4\
 --seed 0\
 --num_workers 5\
 --lr 0.0001\
 --name vae_64_sigma_down_latent_11 --checkpoints_dir $HOME/challenging-nodes/crfill/checkpoints/vae_64_sigma_down_latent_11 --gpu_ids 3