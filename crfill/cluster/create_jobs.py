import numpy as np
import math

from typing import Union, Iterable

training_script = "train_vae.py"
gpus_per_task = 1

# defaults
variable_params = {
    "batchSize": 64,
    "niter_decay": 500,
    "niter": 1000,
    "sigma": 0.1,
    "beta_kl": 1,
    "latent_size": 64,
    "downsample": 2,
    "seed": 0,
    "num_workers": 5,
    "lr": 0.0001,
}

standard_params = (
    f"python -u {training_script} --dataset_mode_train custom_train_vae --dataset_mode custom_train_vae "
    f"--train_image_dir /data --preprocess_mode none --validation_freq 50000 "
    f"--display_freq 10000 --model vae"
)

jobs_source = "$HOME/challenging-nodes/crfill"
singularity_image = "$HOME/crfill.sif"
data = "$TMPDIR/spapa"
training_script = "train_vae.py"


def create_range(
    min: Union[int, float],
    max: Union[int, float],
    num_steps: int,
    distr: str = "uniform",
) -> Iterable:
    if distr == "uniform":
        values = []
        for i in range(num_steps):
            values.append(min + (max - min) / (num_steps - 1) * i)

        return values


ranges = {"sigma": ("exponential", -3, 1), "beta_kl": ("exponential", -3, 1)}

from itertools import product

num_tasks = 16

rng = np.random.default_rng(0)

params_combinations = {"task_id": list(range(num_tasks))}

for param, (range_type, min, max) in ranges.items():
    params_combinations[param] = []
    for i in range(num_tasks):
        if range_type == "exponential":
            exponent = rng.uniform(min, max)
            params_combinations[param].append(10 ** exponent)

gpus_per_node = 4

lines = None

experiment_name = "vae_sigma_beta_param_search"
from pathlib import Path

Path(experiment_name).mkdir(exist_ok=True)
for i in range(num_tasks):
    if lines is None:
        lines = []
        with open("template.sh", "r") as f:
            template_lines = f.readlines()

        for line in template_lines:
            lines.append(
                line.replace("experiment_name", f"{experiment_name}_{i//gpus_per_node}")
            )
    current_params = standard_params

    for param in variable_params:
        if param in params_combinations:
            cur = params_combinations[param][i]
        else:
            cur = variable_params[param]

        current_params += f" --{param} {cur}"
    current_params += f" --name {experiment_name}_{i}"
    current_params += f" --checkpoints_dir $HOME/challenging-nodes/crfill/checkpoints/{experiment_name}_{i}"
    current_params += f" --gpu_ids {i%gpus_per_node}"
    lines.append("\n")

    singularity_command = (
        f"singularity exec --no-home --nv "
        f"--bind {data}:/data "
        f"--bind {jobs_source} "
        f"--bind $HOME/challenging-nodes/crfill/checkpoints "
        f"--pwd {jobs_source} "
        f"{singularity_image} "
        f"{current_params}"
    )
    lines.append(singularity_command)

    if ((i + 1) % gpus_per_node == 0) or (i == (num_tasks - 1)):
        with open(
            Path(experiment_name) / Path(f"job_{(i//gpus_per_node)}.sh"), "w+"
        ) as f:
            f.writelines(lines)
        lines = None
    else:
        # to run all the commands in the same node
        lines[-1] += " &"
