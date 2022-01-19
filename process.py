import SimpleITK
import numpy as np

from evalutils import SegmentationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)
from utils import *
from typing import Dict
import json
from skimage.measure import regionprops
import imageio
from pathlib import Path
import time
import pandas as pd
import random
from random import randrange
import os

import pickle
import torch

from model_submission.trainer.vae_trainer import BaselineVAEOpts
from model_submission.model.vae_model import vaemodel

import matplotlib.pyplot as plt

# This parameter adapts the paths between local execution and execution in docker. You can use this flag to switch between these two modes.
# For building your docker, set this parameter to True. If False, it will run process.py locally for test purposes.
execute_in_docker = False


class Nodulegeneration(SegmentationAlgorithm):
    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
            input_path = Path("/input/") if execute_in_docker else Path("./test/"),
            output_path = Path("/output/") if execute_in_docker else Path("./output/"),
            output_file = Path("/output/results.json") if execute_in_docker else Path("./output/results.json")

        )

        # load nodules.json for location
        with open("/input/nodules.json" if execute_in_docker else "test/nodules.json") as f:
            self.data = json.load(f)


        # ---- load model
        vaeopts = BaselineVAEOpts.copy()
        # vaeopts["trainer"]["optimizer_config"]["lr"] = opt.lr
        # vaeopts["model"]["latent_size"] = opt.latent_size
        # vaeopts["model"]["sigma"] = opt.sigma
        # vaeopts["model"]["beta_kl"] = opt.beta_kl
        # vaeopts["model"]["encoder_params"]["downsample"] = opt.downsample

        vaeModel = vaemodel(**vaeopts["model"])
        vaeModel.cuda()
        vaeModel.load()

        self.netG = vaeModel

    @staticmethod
    def upscale_lesion(lesion, lesion_bbox):
        bbox_height = lesion_bbox[3].item()
        bbox_wid = lesion_bbox[2].item()
        lesion_height = lesion.shape[0]
        lesion_width = lesion.shape[1]
        scale_factor_height = bbox_height / lesion_height
        scale_factor_width = bbox_wid / lesion_width
        scale_factor = min(scale_factor_width, scale_factor_height)
        upscale = ndi.interpolation.zoom(lesion, scale_factor, mode='nearest', order=3)
        upscale = upscale * 0.4
        return upscale

    def place_addition_on_cxr(self, lesion, starting_cxr, lesion_bbox):
        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return idx

        file_to_read = open("model_submission/metadata.pkl", "rb")
        db = pickle.load(file_to_read)

        xcoords = np.array(db['dim0'])
        ycoords = np.array(db['dim1'])

        aspect_ratios = xcoords / ycoords
        inverse_ratios = ycoords / xcoords
        areas = xcoords * ycoords

        ratios = (lesion_bbox[3] / lesion_bbox[2]).detach().cpu().numpy()

        batch_size = starting_cxr.shape[0]
        for sample in range(batch_size):
            idx = find_nearest(aspect_ratios, ratios[sample])
            inverse_idx = find_nearest(inverse_ratios, ratios[sample])
            inverse = False
            if areas[idx] > areas[inverse_idx]:
                sel_idx = idx
            else:
                sel_idx = inverse_idx
                inverse = True

            # TODO: initialize whatever lesion_paths is in init
            # TODO: also find_lesion
            image_path = self.dataset.lesions_paths[sel_idx]
            loaded_lesion = torch.Tensor(np.load(image_path)['arr_0']).unsqueeze(0).unsqueeze(0).cuda()
            lesion = self.netG.sample(x=loaded_lesion, samples=10)
            bbox, crop_lesion = self.find_lesion(lesion)

            new_a_ratios = bbox

            plt.imshow(loaded_lesion.reshape(loaded_lesion.shape[-2:]))
            # draw_rectangle(lesion_bbox.cpu().numpy()[0])
            plt.show()

            # lesion = self.netG.sample(samples=starting_cxr.shape[0], change_dim=13, change_val=.98)

        starting_cxr = (starting_cxr + 1) / 2
        addition = convert_to_range_0_1(lesion)

        new = addition * 255
        new = new.squeeze().detach().cpu().numpy().astype('uint8')
        mask_lesions = torch.zeros_like(addition)
        generated_bbox = []
        scaled_down_lesions = []

        for sample in range(mask_lesions.shape[0]):
            result = cv2.GaussianBlur(new[sample], (11, 11), 5)
            result = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            result = 255 - result
            non_zero = np.argwhere(result)
            top_left = non_zero.min(axis=0)
            bottom_right = non_zero.max(axis=0)
            generated_bbox.append((top_left[0], top_left[1], bottom_right[0], bottom_right[1]))
            result = addition[sample][:, top_left[0]:bottom_right[0] + 1,
                     top_left[1]:bottom_right[1] + 1].detach().cpu().numpy()
            scaled_down_lesions.append(result.reshape(result.shape[-2:]))

        results = np.zeros_like(starting_cxr.detach().cpu().numpy())
        for sample in range(results.shape[0]):
            min_row = lesion_bbox[sample][1].item()
            min_col = lesion_bbox[sample][0].item()
            rows = lesion_bbox[sample][3].item()
            cols = lesion_bbox[sample][2].item()
            max_row = min_row + rows
            max_col = min_col + cols
            upscaled_lesion = self.upscale_lesion(scaled_down_lesions[sample], lesion_bbox[sample])
            # input_0 = scaled_down_lesions[sample]
            input_1 = starting_cxr[sample].detach().cpu().numpy().reshape((starting_cxr[sample].shape[-2:]))
            results[sample] = (1 / 255) * poisson_blend(upscaled_lesion, input_1, min_col, max_col, min_row, max_row)
            # results[sample] = torch.Tensor(poisson_edit(addition[sample].permute(1, 2, 0).detach().cpu().numpy(),
            #                                starting_cxr[sample].permute(1, 2, 0).detach().cpu().numpy(),
            #                                mask_lesion[sample].permute(1, 2, 0).detach().cpu().numpy(), (0, 0))).permute(2,0,1)

        # f, axarr = plt.subplots(1,4)

        # plt.imshow(results[0].reshape(results[0].shape[-2:]))
        # draw_rectangle(lesion_bbox.cpu().numpy()[0])
        # plt.show()

        return 2 * (results - 0.5)

    def predict(self, *, input_image: SimpleITK.Image) -> SimpleITK.Image:
        input_image = SimpleITK.GetArrayFromImage(input_image)
        total_time = time.time()
        if len(input_image.shape)==2:
            input_image = np.expand_dims(input_image, 0)
        
        pd_data = pd.read_csv('/opt/algorithm/ct_nodules.csv' if execute_in_docker else "ct_nodules.csv")
        
        nodule_images = np.zeros(input_image.shape)
        
        for j in range(len(input_image)):
            t = time.time()
            cxr_img_scaled = input_image[j,:,:]
            nodule_data = [i for i in self.data['boxes'] if i['corners'][0][2]==j]

            for nodule in nodule_data:
                cxr_img_scaled = convert_to_range_0_1(cxr_img_scaled)
                boxes = nodule['corners']
                # no spacing info in GC with 3D version
                #x_min, y_min, x_max, y_max = boxes[2][0]/spacing_x, boxes[2][1]/spacing_y, boxes[0][0]/spacing_x, boxes[0][1]/spacing_y
                y_min, x_min, y_max, x_max = boxes[2][0], boxes[2][1], boxes[0][0], boxes[0][1]

                x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

                #------------------------------ Randomly choose ct patch and scale it according to bounding box size.
                required_diameter = max(x_max-x_min, y_max-y_min)
                ct_names = pd_data[pd_data['diameter']>int((required_diameter/5))]['img_name'].values
                if len(ct_names)<1:
                    pd_data[pd_data['diameter']>int((required_diameter/10))]['img_name'].values
                    
                index_ct = random.randint(0, len(ct_names)-1)
                path_nodule = '/opt/algorithm/nodule_patches/' if execute_in_docker else 'nodule_patches/'
                X_ct_2d_resampled, diameter = process_CT_patches(os.path.join(path_nodule,ct_names[index_ct]), os.path.join(path_nodule, ct_names[index_ct].replace('dcm','seg')), required_diameter)
                
                crop = cxr_img_scaled[x_min:x_max, y_min:y_max].copy()
                new_arr = convert_to_range_0_1(X_ct_2d_resampled)

                # contrast matching:
                c = contrast_matching(new_arr, cxr_img_scaled[x_min:x_max, y_min:y_max])
                nodule_contrasted = new_arr * c

                indexes = nodule_contrasted!=np.min(nodule_contrasted)
                result = poisson_blend(nodule_contrasted, cxr_img_scaled, y_min, y_max, x_min, x_max)
                result[x_min:x_max, y_min:y_max] = np.mean(np.array([crop*255, result[x_min:x_max, y_min:y_max]]), axis=0)
                cxr_img_scaled = result.copy()

            nodule_images[j,:,:] = result 
        print('total time took ', time.time()-total_time)
        return SimpleITK.GetImageFromArray(nodule_images)

if __name__ == "__main__":
    Nodulegeneration().process()

                                                                 
                                                                 
