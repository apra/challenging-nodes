# challenging-nodes

##Generative Inpainting Model
Example on a test-image that is provided can be called as:
```
python .\torch_generative_inpainting\test_single.py 
--image .\torch_generative_inpainting\examples\imagenet\imagenet_patches_ILS
VRC2012_val_00000827_input.png 
--checkpoint_path .\torch_generative_inpainting\checkpoints\hole_benchmark\
```

where we add some flags for the image to be painted and the model to use (checkpoint-path).
It will standardly save as output.png in home folder.


For the tensorflow model: 
```
 python .\torch_generative_inpainting\test_tf_model.py --image .\torch_generative_inpainting\examples\imagenet\imagenet_patches_I
LSVRC2012_val_00000827_input.png --model-path .\torch_generative_inpainting\checkpoints\tf_converted\torch_model.p --mask .\torch_generative_inpainting\examples\center_mask_256.png
```

##CRFill Model
For the CRFILL model, follow the instructions in their README (create the environment according to the environment.yml file) then run the following script:
```console
python test.py --batchSize 1 --nThreads 1 --name objrmv --dataset_mode testimage --image_dir ./datasets/places2sample1k_val/places2samples1k_crop256 --mask_dir ./datasets/places2sample1k_val/places2samples1k_256_mask_square128 --output_dir ./results --model inpaint --netG baseconv --which_epoch latest --load_baseg --nThreads 0
```

Here the important thing is to put `--nThreads 0` as to avoid the data loader to fail.