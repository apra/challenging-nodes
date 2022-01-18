# Running the lesion placing
```shell
python train_lesion.py
--network_path
C:\Users\s.papa\docs\code\challenging-nodes\crfill\checkpoints\vae_64\vae_64_sigma_down_latent_16\vae_64_sigma_down_latent_16\latest_net_G.pth
--dataset_mode_train
custom_train_place_lesion
--dataset_mode
custom_train_place_lesion
--model
arrangeplacelesion
--name
vae_64_sigma_down_latent_16
--checkpoints_dir
C:\Users\s.papa\docs\code\challenging-nodes\crfill\checkpoints\vae_64\vae_64_sigma_down_latent_16
--train_image_dir
../../data/data_node21
--train_lesion_dir
../../data/dataset_node21/ct_patches/projected/images
--validation_freq
1000
--gpu_ids
0
--niter
50
--batchSize
4
--display_freq
1
--load_from_checkpoint
--num_workers=0
--netD
deepfill
--node21_resample_count
10
--include_chexpert
--include_mimic
--D_steps_per_G
100000
--no_l1_loss
```

The `--load_from_checkpoint` is especially essential because without 
it you risk over-writing the options `.pkl` or the weights of the trained
VAE. The options `.pkl` is important that it stays the same because 
it needs it to initialize the VAE model correctly.

The outputs are basically just seen in the tensorboard that gets added
in the `/tensorboard/` folder.

