@echo off

SET BSIZE=96
SET BSIZE0=%undefined%
SET %undefined%NWK=16
SET %undefined%PREFIX="--dataset_mode_train trainimage \
--gpu_ids 0,1 \
--name debug \
--dataset_mode_val valimage \
--train_image_dir %CD%\datasets\places\places2 \
--train_image_list %CD%\datasets\places\train_example.txt \
--path_objectshape_list %CD%\datasets\object_shapes.txt \
--path_objectshape_base %CD%\datasets\object_masks \
--val_image_dir %CD%\datasets\places2sample1k_val\places2samples1k_crop256 \
--val_image_list %CD%\datasets\places2sample1k_val\files.txt \
--val_mask_dir %CD%\datasets\places2sample1k_val\places2samples1k_256_mask_square128 \
--no_vgg_loss \
--no_ganFeat_loss \
--load_size 640 \
--crop_size 256 \
--model inpaint \
--netG baseconv \
--netD deepfill \
--preprocess_mode scale_shortside_and_crop \
--validation_freq 10000 \
--gpu_ids 0,1 \
--niter 50 "
python train.py %PREFIX% --batchSize %BSIZE0% --nThreads %NWK% --no_fine_loss --update_part coarse --no_gan_loss --freeze_D --niter 1 %EXTRA%
python train.py %PREFIX% --batchSize %BSIZE% --nThreads %NWK% --update_part fine --continue_train --niter 2 %EXTRA%
python train.py %PREFIX% --batchSize %BSIZE% --nThreads %NWK% --update_part all --continue_train --niter 4 %EXTRA%
SET %EXTRA%PREFIX="--dataset_mode_train trainimage \
--name debugarr0 \
--gpu_ids 0,1 \
--dataset_mode_val valimage \
--train_image_dir %CD%\datasets\places\places2 \
--train_image_list %CD%\datasets\places\train_example.txt \
--path_objectshape_list %CD%\datasets\object_shapes.txt \
--path_objectshape_base %CD%\datasets\object_masks \
--val_image_dir %CD%\datasets\places2sample1k_val\places2samples1k_crop256 \
--val_image_list %CD%\datasets\places2sample1k_val\files.txt \
--val_mask_dir %CD%\datasets\places2sample1k_val\places2samples1k_256_mask_square128 \
--no_vgg_loss \
--no_ganFeat_loss \
--gpu_ids 0,1 \
--load_size 640 \
--crop_size 256 \
--model arrange \
--netG twostagend \
--baseG baseconv \
--norm_type 1 \
--netD deepfill \
--load_base_g %CD%\checkpoints\debug\latest_net_G.pth \
--load_base_d %CD%\checkpoints\debug\latest_net_D.pth \
--lambda_ref 0.5 \
--lambda_l1 1 \
--preprocess_mode scale_shortside_and_crop"
python train.py %PREFIX% --batchSize %BSIZE0% --nThreads %NWK% --update_part all --niter 10 %EXTRA%