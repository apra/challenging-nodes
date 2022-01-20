import opencxr
from opencxr.utils.file_io import read_file, write_file
from image_preprocessing.io_utils import read_paths_for_extension


def preprocess_single(input_dir, input_name, output_dir=""):
    preprocess_algo = opencxr.load(opencxr.algorithms.cxr_standardize)
    # read a file (supports dcm, mha, mhd, png)

    full_cxr_file_path = input_dir + "/" + input_name
    img_np, spacing, _ = read_file(full_cxr_file_path)
    # Do standardization of intensities, cropping to lung bounding box, and resizing to 1024
    std_img, new_spacing, size_changes = preprocess_algo.run(img_np, spacing)
    if output_dir:
        # write the standardized file to disk
        output_cxr_loc = output_dir + "/" + input_name
        write_file(output_cxr_loc, std_img, new_spacing)
    return std_img


def preprocess_folder(base_dir, output_dir, extension=".mha"):
    """
    Preprocesses all the files in (subfolders of) base_dir that have the specified extension. Outputs are saved in output_dir with
    same name as the input image has.
    """
    paths = read_paths_for_extension(base_dir=base_dir, extension=extension)
    for path in paths:
        image_dir = str(path.parent)
        image_name = str(path.name)
        preprocess_single(image_dir, image_name, output_dir)
