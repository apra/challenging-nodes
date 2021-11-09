import numpy as np
from opencxr.utils.file_io import read_file
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from cv2 import cv2
import csv
from image_preprocessing.preprocessor import preprocess_single

# location of metadata file with bboxes corresponding to imagenames
METADATA_LOCATION = 'C:\\Users\\e.marcus\\datasets\\node21\\cxr_images\\proccessed_data\\metadata.csv'


def draw_rectangle(coords, color='r'):
    """Draws a rectangle on inputted [x, y, w, h] coordinates"""
    rect = patches.Rectangle((coords[0], coords[1]), coords[2], coords[3], linewidth=1, edgecolor=color, facecolor='none')
    plt.gca().add_patch(rect)
    plt.show(block=False)


def plot_bounding_boxes(image, bbox_coords, change_bgr_to_rgb=False):
    """Plot image along with one or more bounding boxes inputted as (array of) [x, y, w, h]"""
    if change_bgr_to_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)

    bbox_coords = np.array(bbox_coords)
    if bbox_coords.shape == (4,):
        draw_rectangle(bbox_coords)
    else:
        for bbox in bbox_coords:
            draw_rectangle(bbox)
    plt.show()


def plot_mha_scan(image_path: Path, preprocess_opencxr=False, include_metadata_bbox=True, metadata_location=METADATA_LOCATION):
    """
    Plot a .mha file, can be preprocessed with opencxr, bboxes of nodules can be included by metadata.
    Args:
        image_path: pathlib.Path to image location
        preprocess_opencxr: boolean to follow the node21 preprocess pipeline -- don't use this on already preprocessed files of course
        include_metadata_bbox: boolean for including bboxes found in the metadata file
        metadata_location: location of the metadata file with the bboxes
    Examples:
        >impath = Path('C:\\Users\\e.marcus\\datasets\\node21\\cxr_images\\proccessed_data\\images\\n0373.mha')
        >plot_mha_scan(impath)
    """
    image_dir = str(image_path.parent)
    image_name = str(image_path.name)
    if preprocess_opencxr:
        img_np = preprocess_single(image_dir, image_name)
    else:
        img_np, _, _ = read_file(str(image_path))

    if include_metadata_bbox:
        bboxes = _metadata_extractor(image_name, metadata_location)
        plot_bounding_boxes(img_np, bboxes)
    else:
        plt.imshow(img_np)
        plt.show()


def _metadata_extractor(image_name, metadata_location=METADATA_LOCATION):
    """Reads whole csv to find image_name, so don't use on extremely large csv"""
    bboxes = []
    with open(metadata_location) as f_obj:
        reader = csv.reader(f_obj, delimiter=',')
        for line in reader:  # Iterates through the rows of your csv
            if image_name in str(line):  # If the string you want to search is in the row
                _, h, _, _, w, x, y = [int(entry) if entry.isnumeric() else entry for entry in line]
                bboxes.append([x, y, w, h])
    return bboxes
