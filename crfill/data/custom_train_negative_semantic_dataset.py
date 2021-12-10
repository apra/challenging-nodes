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
from data.custom_transformations import mask_image, crop_around_mask_bbox, normalize_cxr, mask_convention_setter, create_random_bboxes, k_means_image
from util.metadata_utils import get_paths_negatives


class CustomTrainNegativeSemanticDataset(BaseDataset):
    """Custom dataset class for negative xrays -- no nodules. Expects that within the main datafolder there exists a folder
       'negative', that contains negative images in any subdirectory structure. If metadata.csv exists in 'negative' it will read it,
       otherwise it will be created as well."""

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--train_image_dir', type=str, required=True,
                            help='path to the directory that contains photo images')
        parser.add_argument('--train_image_postfix', type=str, default=".mha",
                            help='image extension')
        parser.add_argument('--crop_around_mask_size', type=int, default=256,
                            help='size of the cropped image')
        parser.add_argument('--fold', type=int, default=0,
                            help='current fold to be selected for heldout validation')
        parser.add_argument('--num_folds', type=int, default=10,
                            help='number of folds for the validation')
        return parser

    def initialize(self, opt, paths, mod):
        self.opt = opt
        self.mod = mod
        self.paths = paths
        size = len(self.paths)
        self.full_dataset_size = size
        self.clusters = opt.k_means_clusters
        # TODO: check whether below the right amount of bboxes are created -- in combination with the k-fold
        self.bboxes = create_random_bboxes(number_of_bboxes=self.full_dataset_size, max_x=opt.load_size, max_y=opt.load_size)  # TODO make reproducible RNG here
        self.fold_size = int(self.full_dataset_size / opt.num_folds)
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
        #input image (real images)
        image_path = ''
        index = self.get_true_index(index)
        try:
            image_path = self.paths[index]
            image_mask_bbox = self.bboxes[index]
            full_image = self.mha_loader(image_path)
            crop_size = self.opt.crop_around_mask_size

            image_mask_bbox = mask_convention_setter(
                image_mask_bbox)  # use this method to set conventions for mask bbox
            cropped_image, new_mask_bbox, _ = crop_around_mask_bbox(full_image, image_mask_bbox,
                                                                 crop_size=crop_size, rng=self.rng)  # Crop around nodule
            full_image = np.array(normalize_cxr(full_image), dtype='float32')
            cropped_image = np.array(normalize_cxr(cropped_image), dtype='float32')  # divide 4095
            k_meaned_image, mask_array = k_means_image(cropped_image, new_mask_bbox, clusters=self.clusters)

            crop_bbox = [image_mask_bbox[0] - new_mask_bbox[0], image_mask_bbox[1] - new_mask_bbox[1], crop_size, crop_size]

            mask_tensor = torch.Tensor(mask_array)
            full_image_crop_bbox = torch.Tensor(crop_bbox)
            full_image_bbox = torch.Tensor([image_mask_bbox[0], image_mask_bbox[1], image_mask_bbox[0]+image_mask_bbox[2], image_mask_bbox[1]+image_mask_bbox[3]])
            full_image_tensor = torch.unsqueeze(torch.Tensor(full_image), 0)  # don't self.transform this -- rcnn does its own normalization

            #old:
            #image_tensor = self.transform(cropped_image)  # in this class we don't overlay a mask on the image

            #new:
            image_tensor = self.transform(k_meaned_image)
            image_tensor_org = self.transform(cropped_image)  # in this class we don't overlay a mask on the image

            input_dict = {
                'full_image_bbox': full_image_bbox.float(),  # needed for faster rcnn
                #'full_image_mask_tensor': full_image_mask_tensor.float(),
                'full_image_crop_bbox': full_image_crop_bbox.float(),  # for overlaying the generated result later on
                'full_image': full_image_tensor.float(),  # needed for faster rcnn
                'image_bbox': image_mask_bbox,
                'real_image': image_tensor_org.float(),
                'inputs': image_tensor.float(),
                'mask': mask_tensor.float(),
            }
            return input_dict
        except FileNotFoundError:
            print(f"No image found at: {image_path}")
            return self.__getitem__((index + 1) % self.__len__())
