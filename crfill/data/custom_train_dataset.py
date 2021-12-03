import torchvision.transforms

from data.base_dataset import get_params, get_transform, BaseDataset, basic_transform
from PIL import Image
import os
import pdb
import torch
import numpy as np
import csv
import SimpleITK as sitk
from torchvision.transforms import Compose, ToTensor
from data.custom_transformations import mask_image, crop_around_mask_bbox, normalize_cxr, mask_convention_setter
from util.metadata_utils import get_paths_and_nodules


class CustomTrainDataset(BaseDataset):
    """Custom dataset class for positive xrays, folder structure should consist of main data folder with subfolders 'node21',
       'chexpert' and 'mimic'. Each of these folders should contain their respective metadata.csv file. (for node21, I expect only
       the images folder found inside of cxr_imaged/processed_data)"""

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--train_image_dir', type=str, required=True,
                            help='path to the directory that contains photo images')
        parser.add_argument('--train_image_postfix', type=str, default=".mha",
                            help='image extension')
        parser.add_argument('--crop_around_mask_size', type=int, default=256,
                            help='size of crop around the mask, default=256')
        parser.add_argument('--fold', type=int, default=0,
                            help='current fold to be selected for heldout validation')
        parser.add_argument('--num_folds', type=int, default=10,
                            help='number of folds for the validation')
        parser.add_argument('--include_chexpert', action='store_true',
                            help='Include chexpert positive-lesion dataset')
        parser.add_argument('--include_mimic', action='store_true',
                            help='Include mimic positive-lesion dataset')
        parser.add_argument('--node21_resample_count', type=int, default=0,
                            help='How many times node21 data is resampled')
        return parser

    def initialize(self, opt, path_and_nodules, mod):
        self.opt = opt
        self.mod = mod
        self.paths_and_nodules = path_and_nodules

        self.full_dataset_size = len(self.paths_and_nodules)
        self.fold_size = int(self.full_dataset_size / opt.num_folds)
        if self.full_dataset_size > 10000:
            self.fold_size = 200
        self.begin_fold_idx = opt.fold * self.fold_size
        if opt.fold == opt.num_folds - 1:
            # this is the last fold, take all the remaining samples
            self.end_fold_idx = self.full_dataset_size - 1
            self.fold_size = self.full_dataset_size - self.begin_fold_idx
        else:
            self.end_fold_idx = self.begin_fold_idx + self.fold_size - 1

        if self.mod == 'train':
            self.dataset_size = self.full_dataset_size - self.fold_size
        elif self.mod == 'valid':
            self.dataset_size = self.fold_size

        self.transform = basic_transform()

        self.rng = np.random.default_rng(seed=opt.seed)

    def get_true_index(self, index):
        if self.mod == "train":
            if index < self.begin_fold_idx:
                return index
            else:
                return index + self.end_fold_idx
        elif self.mod == "valid":
            return self.begin_fold_idx + index

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
        # TODO make this process much faster, remove all useless checks
        # input image (real images)
        image_path = ''
        index = self.get_true_index(index)
        try:
            image_path = self.paths_and_nodules[index][0]
            image_mask_bbox = self.paths_and_nodules[index][1]
            full_image = self.mha_loader(image_path)
            crop_size = self.opt.crop_around_mask_size

            # Crop around nodule
            cropped_image, new_mask_bbox = crop_around_mask_bbox(full_image, image_mask_bbox,
                                                                 crop_size=crop_size,
                                                                 rng=self.rng)
            cropped_image = normalize_cxr(cropped_image)  # divide 4095
            cropped_masked_image, mask_array = mask_image(cropped_image, new_mask_bbox)

            # params = get_params(self.opt, cropped_image.shape)

            mask_tensor = torch.Tensor(mask_array)
            image_tensor = self.transform(cropped_image)
            masked_image_tensor = self.transform(cropped_masked_image)
            input_dict = {
                'original_image': torch.Tensor(full_image),
                'image_bbox': image_mask_bbox,
                'real_image': image_tensor.float(),
                'inputs': masked_image_tensor.float(),
                'mask': mask_tensor.float()
            }
            return input_dict
        except FileNotFoundError:
            print(f"No image found at: {image_path}")
            return self.__getitem__((index + 1) % self.__len__())
