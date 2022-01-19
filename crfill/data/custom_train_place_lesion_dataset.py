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
import torch.utils.data as data


class CustomTrainPlaceLesionDataset(BaseDataset):
    """Custom dataset class for negative xrays -- no nodules. Expects that within the main datafolder there exists a folder
       'negative', that contains negative images in any subdirectory structure. If metadata.csv exists in 'negative' it will read it,
       otherwise it will be created as well."""

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--train_lesion_dir', type=str, required=True,
                            help='path to the directory that contains projected lesion images')
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
        parser.add_argument('--include_chexpert', action='store_true',
                            help='Include chexpert positive-lesion dataset')
        parser.add_argument('--include_mimic', action='store_true',
                            help='Include mimic positive-lesion dataset')
        parser.add_argument('--node21_resample_count', type=int, default=0,
                            help='How many times node21 data is resampled')
        return parser

    def initialize(self, opt, paths, mod, metadata=None):
        self.opt = opt
        self.rng = np.random.default_rng(seed=opt.seed)
        self.mod = mod
        positive_paths,negative_paths, lesion_paths = paths
        self.lesions_paths = lesion_paths
        self.metadata = metadata

        metadata['aspect_ratio'] = []
        metadata['inverse_aspect_ratio'] = []

        for i in range(len(self.metadata['dim0'])):
            metadata['aspect_ratio'].append(self.metadata['dim0'][i]/self.metadata['dim1'][i])
            metadata['inverse_aspect_ratio'].append(self.metadata['dim1'][i] / self.metadata['dim0'][i])

        if len(negative_paths) < len(positive_paths):
            print("There are more negative samples than positive ones.")
            positive_paths = positive_paths[:len(negative_paths)]
        else:
            negative_paths = negative_paths[:len(positive_paths)]

        self.positive_paths = positive_paths
        self.negative_paths = negative_paths
        self.full_dataset_size = len(self.positive_paths)
        self.negative_bboxes = create_random_bboxes(number_of_bboxes=self.full_dataset_size, max_x=opt.load_size,
                                                    max_y=opt.load_size,
                                                    rng=self.rng)  # TODO make reproducible RNG here
        # TODO: check whether below the right amount of bboxes are created -- in combination with the k-fold
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

    def get_true_index(self, index):
        if self.mod == "train":
            if index < self.begin_fold_idx:
                return index
            else:
                return index + self.end_fold_idx
        elif self.mod == "valid":
            return self.begin_fold_idx + index

    @staticmethod
    def mha_loader(image_path):
        img = sitk.ReadImage(image_path, imageIO="MetaImageIO")
        img_np = sitk.GetArrayFromImage(img)
        return img_np, img.GetSpacing()

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        pos_image_path = ''
        neg_image_path = ''
        index = self.get_true_index(index)
        try:
            pos_image_path = self.positive_paths[index][0]
            neg_image_path = self.negative_paths[index]

            pos_lesion_bbox = self.positive_paths[index][1]
            neg_lesion_bbox = self.negative_bboxes[index]

            pos_cxr = normalize_cxr(self.mha_loader(pos_image_path)[0])
            orig_img, spacing = self.mha_loader(neg_image_path)
            neg_cxr = normalize_cxr(orig_img)

            crop_size = self.opt.crop_around_mask_size

            pos_cropped_cxr, pos_cropped_lesion_bbox, pos_crop_bbox = crop_around_mask_bbox(pos_cxr,
                                                                                            pos_lesion_bbox,
                                                                                            crop_size=crop_size,
                                                                                            rng=self.rng)

            pos_cropped_masked_cxr, pos_cropped_mask = mask_image(pos_cropped_cxr, pos_cropped_lesion_bbox)

            pos_cropped_normalized_cxr = self.transform(pos_cropped_cxr)

            neg_cropped_cxr, neg_cropped_lesion_bbox, neg_crop_bbox = crop_around_mask_bbox(neg_cxr,
                                                                                            neg_lesion_bbox,
                                                                                            crop_size=crop_size,
                                                                                            rng=self.rng)

            neg_cropped_masked_cxr, neg_cropped_mask = mask_image(neg_cropped_cxr, neg_cropped_lesion_bbox)

            neg_cropped_normalized_cxr = self.transform(neg_cropped_cxr)

            neg_cxr = torch.unsqueeze(torch.Tensor(neg_cxr), 0)

            full_image_bbox = torch.Tensor([neg_lesion_bbox[0], neg_lesion_bbox[1], neg_lesion_bbox[0]+neg_lesion_bbox[2], neg_lesion_bbox[1]+neg_lesion_bbox[3]])

            input_dict = {
                # the full CXR of the negative sample
                'neg_cxr': neg_cxr.float(),
                # cropped and normalized (between -0.5 and 0.5) CXR of the positive sample (input to the discriminator)
                'pos_cropped_normalized_cxr': pos_cropped_normalized_cxr.float(),
                # cropped and normalized (between -0.5 and 0.5) CXR of the negative sample (input to the generator)
                'neg_cropped_normalized_cxr': neg_cropped_normalized_cxr.float(),
                # array with the [x,y,w,h] of the cropping bounding box in the negative sample (used for placing result)
                'neg_crop_bbox': torch.Tensor(neg_crop_bbox),
                # mask used to select the region where the tumor should be placed by the generator
                'neg_cropped_mask': torch.Tensor(neg_cropped_mask),
                # bbox used for the faster rcnn
                'neg_lesion_bbox': full_image_bbox,
                'neg_spacing': spacing
            }
            return input_dict
        except FileNotFoundError:
            print(f"No image found at: {pos_image_path} or {neg_image_path}")
            return self.__getitem__((index + 1) % self.__len__())
