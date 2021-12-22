import numpy as np
import math

downsample = 2
sigma = 0.1
beta_kl = 1
latent_size = 64
epochs = 1000
decay_epochs = 500
batch_size = 64

total_samples = 1186

total_epochs = epochs + decay_epochs

num_iters_per_epoch = math.ceil(total_samples // batch_size)

standard_params = f"--seed 0 --batchSize {batch_size} --niter_decay {decay_epochs} " \
                  f"--dataset_mode_train custom_train_vae --dataset_mode custom_train_vae " \
                  f"--train_image_dir /data --preprocess_mode none --validation_freq 50000 " \
                  f"--niter {epochs} --display_freq 10000 --model vae"
jobs_source = "$HOME/challenging-nodes/crfill"
singularity_image = "$HOME/crfill.sif"
data = "$TMPDIR/spapa"
training_script = "train_vae.py"

experiment_name = f'vae-dw{downsample}-s{sigma}-l{latent_size}-b{beta_kl}'

with open("template.sh", "r") as f:
    lines = f.readlines()
output_lines = []
for line in lines:
    output_lines.append(line.replace('experiment_name', experiment_name))

with open("job_file.sh", "w+") as f:
    output_lines.append()
    f.writelines(lines)
