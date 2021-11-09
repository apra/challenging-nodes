import os
from pathlib import Path


def read_paths_for_extension(base_dir, extension, save_to_file=False):
    """Function that starts from base_dir and searches all subfolders for the inputted extension and saves the locations to a file.
       Useful for finding, e.g., all images in a certain directory."""
    path_list = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(extension):
                file_location = Path(os.path.join(root, file))
                path_list.append(file_location)

    if save_to_file:
        save_file_name = str(os.path.basename(base_dir)) + "__" + str(extension[1:]) + ".txt"  # create output filename
        with open(save_file_name, "w") as f:
            for path in path_list:
                f.write("%s \n" % path)

    return path_list
