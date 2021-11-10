import numpy as np
from bbox import crop_to_bbox


def mask_image(image: np.ndarray, mask):
    pass


def crop_around_mask(image: np.ndarray, mask, crop_size=256, seed=0):
    """create random bbox of crop_size**2 that includes mask region and stays within image"""
    if len(image.shape) != 2:
        raise ValueError('Image to be cropped is not of shape (x,y) -- input only single channel image')
    im_max_x, im_max_y = image.shape
    mask_x, mask_y, mask_w, mask_h = mask
    if seed:
        np.random.seed(seed)

    crop_min_x = max(mask_x + mask_w - crop_size + 1, 0)
    crop_max_x = min(mask_x - 1, im_max_x - crop_size - 1)
    crop_min_y = max(mask_y + mask_h - crop_size + 1, 0)
    crop_max_y = min(mask_y - 1, im_max_y - crop_size - 1)

    crop_x = np.random.randint(crop_min_x, crop_max_x)
    crop_y = np.random.randint(crop_min_y, crop_max_y)

    cropped_image = crop_to_bbox(image, [crop_x, crop_y, crop_size, crop_size])
    return cropped_image
