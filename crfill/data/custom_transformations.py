import numpy as np
from data.bbox import crop_to_bbox
from typing import List
import cv2
import matplotlib.pyplot as plt


def k_means_image(image: np.ndarray, mask_bbox, mask_value=1, clusters=3):
    [x, y, w, h] = mask_convention_setter(mask_bbox, invert=True)
    mask_image = image[y:y + h, x:x + w].copy()
    mask_image_ref = image.copy()
    mask_image *= 255.
    im = mask_image.reshape((-1, 1))
    imf = np.float32(im)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = clusters
    attempts = 1
    ret, label, center = cv2.kmeans(imf, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((mask_image.shape))
    ref_values = np.unique(res2)

    res2 = np.float32(res2)
    update_value = 0.
    for value in ref_values:
        res2[res2==value] = update_value
        update_value += 0.25

    mask_image_ref[y:y + h, x:x + w] = res2
    mask_array = np.zeros((1, *image.shape))
    mask_array[:, y:y + h, x:x + w] = mask_value

    return mask_image_ref, mask_array


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


def normalize_cxr(image):
    return image / 4095


def mask_convention_setter(mask, invert=False):
    # use this method to change the mask (if we get x,y sequence wrong for example)
    # invert should reverse back to [x,y,w,h]
    if invert:
        return mask
    else:
        return mask


def create_random_bboxes(number_of_bboxes, seed=0, max_x=1024, max_y=1024, rng=None):
    # TODO: make reproducible rng
    if rng is None:
        rng = np.random.default_rng()
    bbox_list = []
    for i in range(number_of_bboxes):
        # distributions that match closely what is found in the data
        l_or_r = rng.random()  # x has this left and right factor because it occurs in lungs, not in between lungs
        if l_or_r > 0.5:  # right lung
            x = min(930, max(rng.normal(725, 80), 530))
        else:  # left lung
            x = max(20, min(rng.normal(225, 80), 450))

        y = (rng.beta(2, 2) + (1 / 7)) * 700  # is bounded [100, 800]
        w = min(rng.gamma(8, 7.5), max_x - x, 230)  # 230 is the max of the simulated_metadata, which we will have to predict on
        h = min(rng.gamma(7, 8.4), max_y - y, 230)
        bbox_list.append([int(x), int(y), int(w), int(h)])
    return bbox_list


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


def flip_image_and_bbox(image: np.ndarray, bbox: List):
    width = image.shape[0]
    flipped_image = np.flip(image, axis=1)
    flipped_bbox = [width - (bbox[0]+bbox[2]), bbox[1], bbox[2], bbox[3]]
    return flipped_image, flipped_bbox