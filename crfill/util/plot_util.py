import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from cv2 import cv2
import csv

# location of metadata file with bboxes corresponding to imagenames
METADATA_LOCATION = (
    "C:\\Users\\e.marcus\\datasets\\node21\\cxr_images\\proccessed_data\\metadata.csv"
)


def draw_rectangle(coords, color="r"):
    """Draws a rectangle on inputted [x, y, w, h] coordinates"""
    rect = patches.Rectangle(
        (coords[0], coords[1]),
        coords[2],
        coords[3],
        linewidth=1,
        edgecolor=color,
        facecolor="none",
    )
    plt.gca().add_patch(rect)
    plt.show(block=False)


def draw_bounding_boxes(image, bbox_coords, change_bgr_to_rgb=False):
    fig = plt.figure()
    fig.add_subplot(111)
    """Plot image along with one or more bounding boxes inputted as (array of) [x, y, w, h]"""
    if change_bgr_to_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image.reshape(1024, 1024))

    bbox_coords = np.array(bbox_coords)
    if bbox_coords.shape == (4,):
        draw_rectangle(bbox_coords)
    else:
        for bbox in bbox_coords:
            draw_rectangle(bbox)

    fig.canvas.draw()
    fig.tight_layout(pad=0)
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data
