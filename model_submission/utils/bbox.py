import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import List, Union
import torch


def crop_to_bbox(
    data: Union[np.ndarray, torch.Tensor], bbox: List[int], pad_value: int = 0
) -> Union[np.ndarray, torch.Tensor]:
    """Extract bbox from images, coordinates can be negative.
    Parameters
    ----------
    data : np.ndarray or torch.tensor
       nD array or torch tensor.
    bbox : list or tuple
       bbox of the form (coordinates, size),
       for instance (4, 4, 2, 1) is a patch starting at row 4, col 4 with height 2 and width 1.
    pad_value : number
       if bounding box would be out of the image, this is value the patch will be padded with.
    Returns
    -------
    ndarray
        Numpy array of data cropped to BoundingBox
    """
    if not isinstance(data, (np.ndarray, torch.Tensor)):
        raise ValueError(f"Expected `data` to be ndarray or tensor. Got {type(data)}.")

    # Coordinates, size
    ndim = len(bbox) // 2
    if len(bbox) % 2 != 0:
        raise ValueError(f"Bounding box should have the form of [x_0, x_1, ..., h_0, h_1], but got length {ndim}.")
    bbox_coords, bbox_size = np.asarray(bbox[:ndim]), np.asarray(bbox[ndim:])
    # Offsets
    l_offset = -bbox_coords.copy()
    l_offset[l_offset < 0] = 0

    r_offset = (bbox_coords + bbox_size) - np.array(data.shape)
    r_offset[r_offset < 0] = 0

    region_idx = [slice(i, j) for i, j in zip(bbox_coords + l_offset, bbox_coords + bbox_size - r_offset)]

    if isinstance(data, torch.Tensor):
        # TODO(jt): Investigate if clone is needed
        out = data[tuple(region_idx)].clone()
    else:
        out = data[tuple(region_idx)].copy()

    if np.all(l_offset == 0) and np.all(r_offset == 0):
        return out

    # If we have a positive offset, we need to pad the patch.
    if isinstance(data, torch.Tensor):
        patch = pad_value * torch.ones(bbox_size.tolist(), dtype=data.dtype)
    else:
        patch = pad_value * np.ones(bbox_size, dtype=data.dtype)

    patch_idx = [slice(i, j) for i, j in zip(l_offset, bbox_size - r_offset)]

    patch[tuple(patch_idx)] = out

    return patch


def mask_image(image: np.ndarray, mask_bbox, mask_value=1, randomized_mask=False, rng=None):
    [x, y, w, h] = mask_convention_setter(mask_bbox, invert=True)  # makes sure bbox is [x,y,w,h]
    mask_image = image.copy()
    if randomized_mask:
        if rng is None:
            rng = np.random.default_rng()
        mask_value = rng.random((h, w))
    mask_image[y:y+h, x:x+w] = mask_value
    mask_array = np.zeros((1, *image.shape))
    mask_array[:, y:y+h, x:x+w] = mask_value
    return mask_image, mask_array


def mask_convention_setter(mask, invert=False):
    # use this method to change the mask (if we get x,y sequence wrong for example)
    # invert should reverse back to [x,y,w,h]
    if invert:
        return mask
    else:
        return mask



def crop_around_mask_bbox(image: np.ndarray, mask_bbox, crop_size=256, rng=None, return_new_mask_bbox=True):
    """create random bbox of crop_size**2 that includes mask region and stays within image"""
    if len(image.shape) != 2:
        raise ValueError('Image to be cropped is not of shape (x,y) -- input only single channel image')

    # mask_bbox = mask_convention_setter(mask_bbox, invert=True)  # this guarantees the mask is [x,y,w,h]

    im_max_x, im_max_y = image.shape
    mask_x, mask_y, mask_w, mask_h = mask_bbox
    if rng is None:
        rng = np.random.default_rng(seed=0)

    crop_min_x = max(mask_x + mask_w - crop_size, 0)
    crop_max_x = min(mask_x, im_max_x - crop_size)
    crop_min_y = max(mask_y + mask_h - crop_size, 0)
    crop_max_y = min(mask_y, im_max_y - crop_size)

    if crop_min_y == crop_max_y:
        crop_y = crop_min_y
    else:
        crop_y = rng.integers(crop_min_y, crop_max_y)

    if crop_min_x == crop_max_x:
        crop_x = crop_min_x
    else:
        crop_x = rng.integers(crop_min_x, crop_max_x)

    cropped_image = crop_to_bbox(image, [crop_y, crop_x, crop_size, crop_size])
    new_mask = [mask_x - crop_x, mask_y - crop_y, mask_w, mask_h]
    new_mask = mask_convention_setter(new_mask)

    assert crop_x + crop_size <= im_max_x, f"Crop_x is {crop_x}, such that we find max x of {crop_x + crop_size}, bbox: {mask_bbox}"
    assert crop_y + crop_size <= im_max_y, f"Crop_y is {crop_y}, such that we find max x of {crop_y + crop_size}"

    if return_new_mask_bbox:
        return cropped_image, new_mask, [crop_x, crop_y, crop_size, crop_size]
    else:
        return cropped_image
