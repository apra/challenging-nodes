import numpy as np
from data.bbox import crop_to_bbox


def mask_image(image: np.ndarray, mask_bbox, mask_value=1):
    [x, y, w, h] = mask_convention_setter(mask_bbox, invert=True)  # makes sure bbox is [x,y,w,h]
    mask_image = image.copy()
    mask_image[y:y+h, x:x+w] = mask_value
    mask_array = np.zeros((1, *image.shape))
    mask_array[:, y:y+h, x:x+w] = mask_value
    return mask_image, mask_array


def normalize_cxr(image):
    return image / 4095


def mask_convention_setter(mask, invert=False):
    # use this method to change the mask (if we get x,y sequence wrong for example)
    # invert should reverse back to [x,y,w,h]
    if invert:
        return mask
    else:
        return mask


def crop_around_mask_bbox(image: np.ndarray, mask_bbox, crop_size=256, seed=0, return_new_mask_bbox=True):
    """create random bbox of crop_size**2 that includes mask region and stays within image"""
    if len(image.shape) != 2:
        raise ValueError('Image to be cropped is not of shape (x,y) -- input only single channel image')

    mask_bbox = mask_convention_setter(mask_bbox, invert=True)  # this guarantees the mask is [x,y,w,h]

    im_max_x, im_max_y = image.shape
    mask_x, mask_y, mask_w, mask_h = mask_bbox
    if seed:
        np.random.seed(seed)

    crop_min_x = max(mask_x + mask_w - crop_size + 1, 0)
    crop_max_x = min(mask_x - 1, im_max_x - crop_size - 1)
    crop_min_y = max(mask_y + mask_h - crop_size + 1, 0)
    crop_max_y = min(mask_y - 1, im_max_y - crop_size - 1)

    crop_x = np.random.randint(crop_min_x, crop_max_x)
    crop_y = np.random.randint(crop_min_y, crop_max_y)

    cropped_image = crop_to_bbox(image, [crop_y, crop_x, crop_size, crop_size])
    new_mask = [mask_x - crop_x, mask_y - crop_y, mask_w, mask_h]
    new_mask = mask_convention_setter(new_mask)
    if return_new_mask_bbox:
        return cropped_image, new_mask
    else:
        return cropped_image
