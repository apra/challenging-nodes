import csv
import os
import random
from pathlib import Path


def is_image_file(filename, extensions='.mha'):
    return any(filename.endswith(extension) for extension in extensions)


def metadata_dict_node21(metadata_location):
    """Reads whole csv to find image_name, creates dict with nonempty bboxes.
       Output:
       Bboxes dictionary with key the img_name and values the bboxes themselves."""
    bboxes = {}
    with open(metadata_location) as f_obj:
        reader = csv.reader(f_obj, delimiter=',')
        for line in reader:
            _, h, img_name, _, w, x, y = [int(entry) if entry.isnumeric() else entry for entry in line]
            if h != 0 and w != 0:  # only append nonempty bboxes
                img_name = str(Path(img_name))  # compatibility between different OS
                bboxes.setdefault(img_name, [])  # these two lines allow safe placing of multiple values for key
                bboxes[img_name].append([x, y, w, h])
    return bboxes


def metadata_dict_chex_mimic(metadata_location):
    """Reads whole csv to find image_name, creates dict with nonempty bboxes
       Output:
       Bboxes dictionary with key the img_name and values the bboxes themselves."""
    bboxes = {}
    with open(metadata_location) as f_obj:
        reader = csv.reader(f_obj, delimiter=',')
        next(reader)  # skip header
        for line in reader:
            _, img_name, x, y, w, h = [int(entry) if entry.isnumeric() else entry for entry in line]
            if h != 0 and w != 0:  # only append nonempty bboxes
                img_name = str(Path(img_name))  # compatibility between different OS
                bboxes.setdefault(img_name, [])  # these two lines allow safe placing of multiple values for key
                bboxes[img_name].append([x, y, w, h])
    return bboxes


def get_paths_and_nodules_helper(image_dir, chex_or_mimic=False):
    """Helper function that walks the dir/subdirs for images that have bboxes in the metadata file."""
    image_nodule_list = []
    nodule_list = os.path.join(image_dir, Path('metadata.csv'))
    if chex_or_mimic:
        metadata_dict = metadata_dict_chex_mimic(nodule_list)
    else:
        metadata_dict = metadata_dict_node21(nodule_list)
    for root, dnames, fnames in sorted(os.walk(image_dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                if chex_or_mimic:
                    image_dir = os.path.join(Path(image_dir), Path(""))  # adds slashes at end of name appropriate to OS
                    image_dir = str(image_dir)[:-1]  # remove '.' at end
                    fname = path.replace(image_dir, "")  # adds to the filename the necessary directories to find the image in the dict
                if fname in metadata_dict:
                    for nodule in metadata_dict[fname]:
                        image_nodule_list.append([path, nodule])
    return image_nodule_list


def get_paths_and_nodules(image_dir, include_chexpert=True, include_mimic=True, resample_count_node21=0):
    """Assumes Image_dir is folder containing subdirs node21, chexpert and mimic, each of the respective folders contains metadata.csv."""
    total_image_nodule_list = []
    node21_image_dir = os.path.join(Path(image_dir), Path('node21'))
    chex_image_dir = os.path.join(Path(image_dir), Path('chexpert'))
    mimic_image_dir = os.path.join(Path(image_dir), Path('mimic'))

    total_image_nodule_list += get_paths_and_nodules_helper(node21_image_dir, chex_or_mimic=False)

    for resample in range(resample_count_node21):  # resample node21
        total_image_nodule_list += total_image_nodule_list

    if include_chexpert:
        total_image_nodule_list += get_paths_and_nodules_helper(chex_image_dir, chex_or_mimic=True)
    if include_mimic:
        total_image_nodule_list += get_paths_and_nodules_helper(mimic_image_dir, chex_or_mimic=True)

    # shuffle before selecting the fold, this should remain consistent as random seed is set at the beginning of train
    random.shuffle(total_image_nodule_list)

    return total_image_nodule_list

