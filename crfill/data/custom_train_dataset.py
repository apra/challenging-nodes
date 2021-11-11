import torchvision.transforms

from data.base_dataset import get_params, get_transform, BaseDataset
from PIL import Image
import os
import pdb
import csv
import SimpleITK as sitk
from torchvision.transforms import Compose, ToTensor
from data.custom_transformations import mask_image, crop_around_mask_bbox, normalize_cxr, mask_convention_setter


def is_image_file(filename, extensions='.mha'):
    return any(filename.endswith(extension) for extension in extensions)


def metadata_nonempty_dict(metadata_location):
    """Reads whole csv to find image_name, creates dict with nonempty bboxes"""
    # NOTE: conventions for the sequence of the mask, e.g. x,y,w,h can be changed later with mask_convention_setter
    # TODO: only works for csv file, implement JSON
    bboxes = {}
    with open(metadata_location) as f_obj:
        reader = csv.reader(f_obj, delimiter=',')
        for line in reader:
            _, h, img_name, _, w, x, y = [int(entry) if entry.isnumeric() else entry for entry in line]
            if h != 0 and w != 0:  # only append nonempty bboxes
                bboxes.setdefault(img_name, [])  # these two lines allow safe placing of multiple values for key
                bboxes[img_name].append([x, y, w, h])
    return bboxes


class CustomTrainImageDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--train_image_dir', type=str, required=True,
                            help='path to the directory that contains photo images')
        parser.add_argument('--train_nodule_list', type=str, required=True,
                            help='path to the file that contains nodule locations')
        parser.add_argument('--train_image_postfix', type=str, default=".mha",
                            help='image extension')
        parser.add_argument('--crop_around_mask_size', type=str, default=256,
                            help='size of crop around the mask, default=256')
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.paths_and_nodules = self.get_paths_and_nodules()
        size = len(self.paths_and_nodules)
        self.dataset_size = size

    def get_paths_and_nodules(self):
        image_dir = self.opt.train_image_dir
        nodule_list = self.opt.train_nodule_list
        image_nodule_list = []
        metadata_dict = metadata_nonempty_dict(nodule_list)
        for root, dnames, fnames in sorted(os.walk(image_dir)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    if fname in metadata_dict:
                        for nodule in metadata_dict[fname]:
                            image_nodule_list.append([path, nodule])
        return image_nodule_list

    @staticmethod
    def mha_loader(image_path, return_spacing=False):
        img = sitk.ReadImage(image_path, imageIO="MetaImageIO")
        img_np = sitk.GetArrayFromImage(img)
        if return_spacing:
            return img_np, img.GetSpacing()
        else:
            return img_np

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        # input image (real images)
        image_path = ''
        try:
            image_path = self.paths_and_nodules[index][0]
            image_mask_bbox = self.paths_and_nodules[index][1]
            full_image = self.mha_loader(image_path)
            crop_size = self.opt.crop_around_mask_size

            image_mask_bbox = mask_convention_setter(image_mask_bbox)  # use this method to set conventions for mask bbox

            cropped_image, new_mask_bbox = crop_around_mask_bbox(full_image, image_mask_bbox, crop_size=crop_size)  # Crop around nodule
            cropped_image = normalize_cxr(cropped_image)  # divide 4095
            cropped_masked_image, mask_array = mask_image(cropped_image, new_mask_bbox)

            #params = get_params(self.opt, cropped_image.shape)
            transform_image = get_transform(self.opt, '')
            mask_tensor = Compose([ToTensor()])(mask_array)
            image_tensor = transform_image(cropped_image)
            masked_image_tensor = transform_image(cropped_masked_image)  #TODO is there any randomness in the transform -- then we get different transforms on original and masked...
            input_dict = {
                          'image': image_tensor,
                          'masked_image': masked_image_tensor,
                          'mask': mask_tensor,
                          'path': image_path,
                          }
            return input_dict
        except FileNotFoundError:
            print(f"skip {image_path}")
            return self.__getitem__((index+1) % self.__len__())
