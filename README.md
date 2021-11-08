# challenging-nodes

Example on a test-image that is provided can be called as:
```
python .\torch-generative-inpainting\test_single.py 
--image .\torch-generative-inpainting\examples\imagenet\imagenet_patches_ILS
VRC2012_val_00000827_input.png 
--checkpoint_path .\torch-generative-inpainting\checkpoints\hole_benchmark\
```

where we add some flags for the image to be painted and the model to use (checkpoint-path).
It will standardly save as output.png in home folder.