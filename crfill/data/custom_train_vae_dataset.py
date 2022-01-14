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
from data.custom_transformations import mask_image, crop_around_mask_bbox, normalize_cxr, mask_convention_setter, \
    create_random_bboxes
from util.metadata_utils import get_paths_negatives
from PIL import Image
from pathlib import Path


class CustomTrainVAEDataset(BaseDataset):
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

    def initialize(self, opt, paths, mod, metadata=None):
        self.opt = opt
        self.mod = mod
        self.paths = paths
        size = len(self.paths)
        self.full_dataset_size = size
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

        self.transform_list = []
        # self.transform_list += [torchvision.transforms.ToTensor()]
        self.metadata = metadata
        self.rng = np.random.default_rng(seed=opt.seed)

    def get_metadata(self, path):
        try:
            index = self.metadata["id"].index(Path(path).name[:-4])
        except:
            print("wtf is going on ")
        return {
            "id": self.metadata["id"][index],
            "dim0": self.metadata["dim0"][index],
            "dim1": self.metadata["dim1"][index]
        }

    def get_true_index(self, index):
        if self.mod == "train":
            if index < self.begin_fold_idx:
                return index
            else:
                return index + self.end_fold_idx
        elif self.mod == "valid":
            return self.begin_fold_idx + index

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        image_path = ''
        index = self.get_true_index(index)
        try:
            image_path = self.paths[index]
            full_image = torch.Tensor(np.load(image_path)['arr_0']).unsqueeze(0)
            meta = self.get_metadata(image_path)
            height = meta['dim0']
            width = meta['dim1']
            if self.rng.binomial(1, .5) > 0.1:
                image_tensor = torch.transpose(full_image, 1, 2)
                height = meta['dim1']
                width = meta['dim0']
            else:
                image_tensor = full_image

            # self.transform_list += [
            #     torchvision.transforms.RandomAffine(degrees=90, fill=float(full_image.min()), interpolation=torchvision.transforms.InterpolationMode.BILINEAR)]
            # transform = torchvision.transforms.Compose(self.transform_list)
            # image_tensor = transform(full_image)
            input_dict = {
                'inputs': image_tensor.float(),
                'height': height,
                'weight': width
            }
            return input_dict
        except FileNotFoundError:
            print(f"No image found at: {image_path}")
            return self.__getitem__((index + 1) % self.__len__())
