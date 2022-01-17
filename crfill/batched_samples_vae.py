from pathlib import Path
import subprocess
from subprocess import check_output

dir = Path("checkpoints/vae_64")

for l in list(dir.glob("vae_64_*")):
    current_dir = l.name
    print(current_dir)
    try:
        check_output(f"python sample_vae.py --network_path C:\\Users\\s.papa\\docs\\code\\challenging-nodes\\crfill\\checkpoints\\vae_64\\{current_dir}\\{current_dir}\\latest_net_G.pth --dataset_mode_train custom_train_vae --dataset_mode custom_train_vae --model vae --name {current_dir} --checkpoints_dir C:\\Users\\s.papa\\docs\\code\\challenging-nodes\\crfill\\checkpoints\\vae_64\\{current_dir} --train_image_dir ../../data/dataset_node21/ct_patches/projected/images --validation_freq 1000 --gpu_ids 0 --niter 50 --batchSize 4 --display_freq 100 --load_from_checkpoint --num_workers=0", shell=True).decode()
    except subprocess.CalledProcessError as e:
        continue