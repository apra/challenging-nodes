import SimpleITK
import numpy as np

from evalutils import SegmentationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)
import matplotlib
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
import matplotlib.pyplot as plt

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
            input_path=Path("/input/") if execute_in_docker else Path("./test/"),
            output_path=Path("/output/") if execute_in_docker else Path("./output/"),
            output_file=Path("/output/results.json")
            if execute_in_docker
            else Path("./output/results.json"),
        )


        # load nodules.json for location
        with open(
            "/input/nodules.json" if execute_in_docker else "test/nodules.json"
        ) as f:
            self.data = json.load(f)

        # load options for the selected trained
        model_name = "vae_64_sigma_down_latent_16"
        with open(f"model_submission/model/{model_name}/{model_name}/opt.pkl", "rb") as f:
            opt = pickle.load(f)
        # ---- load model
        vaeopts = BaselineVAEOpts.copy()
        vaeopts["trainer"]["optimizer_config"]["lr"] = opt.lr
        vaeopts["model"]["latent_size"] = opt.latent_size
        vaeopts["model"]["sigma"] = opt.sigma
        vaeopts["model"]["beta_kl"] = opt.beta_kl
        vaeopts["model"]["encoder_params"]["downsample"] = opt.downsample

        vaeModel = vaemodel(**vaeopts["model"])
        vaeModel.network_path = f"model_submission/model/{model_name}/{model_name}/latest_net_G.pth"
        vaeModel.cuda()
        vaeModel.load()
        vaeModel.eval()
        self.vae = vaeModel

        with open("model_submission/metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)
        self.lesions = np.load("model_submission/all_lesions.npz")["arr_0"]

    @staticmethod
    def upscale_lesion(lesion, lesion_bbox):
        bbox_height = lesion_bbox[3].item()
        bbox_wid = lesion_bbox[2].item()
        lesion_height = lesion.shape[0]
        lesion_width = lesion.shape[1]
        scale_factor_height = bbox_height / lesion_height
        scale_factor_width = bbox_wid / lesion_width
        scale_factor = min(scale_factor_width, scale_factor_height)
        upscale = ndi.interpolation.zoom(lesion, scale_factor, mode="nearest", order=3)
        upscale = upscale * 0.4
        return upscale

    @staticmethod
    def find_lesion(lesion_image):
        addition = convert_to_range_0_1(lesion_image)

        new = addition * 255
        new = new.squeeze().detach().cpu().numpy().astype("uint8")
        generated_bbox = []
        scaled_down_lesions = []

        for sample in range(addition.shape[0]):
            result = cv2.GaussianBlur(new[sample], (11, 11), 5)
            result = cv2.threshold(
                result, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )[1]
            result = 255 - result
            non_zero = np.argwhere(result)
            top_left = non_zero.min(axis=0)
            bottom_right = non_zero.max(axis=0)
            generated_bbox.append(
                (top_left[0], top_left[1], bottom_right[0], bottom_right[1])
            )
            result = (
                addition[sample][
                    :,
                    top_left[0] : bottom_right[0] + 1,
                    top_left[1] : bottom_right[1] + 1,
                ]
                .detach()
                .cpu()
                .numpy()
            )
            scaled_down_lesions.append(result.reshape(result.shape[-2:]))

        return np.array(generated_bbox), scaled_down_lesions

    def place_lesion(self, lesion, starting_cxr, lesion_bbox):
        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return idx
        print(lesion_bbox)
        xcoords = np.array(self.metadata["dim0"])
        ycoords = np.array(self.metadata["dim1"])

        aspect_ratios = xcoords / ycoords
        inverse_ratios = ycoords / xcoords
        areas = xcoords * ycoords

        ratio = (lesion_bbox[3] / lesion_bbox[2])
        area = (lesion_bbox[3] * lesion_bbox[2])
        scaled_area = area/40

        idx = find_nearest(aspect_ratios, ratio)
        inverse_idx = find_nearest(inverse_ratios, ratio)
        inverse = False
        if areas[idx] > areas[inverse_idx]:
            sel_idx = idx
        else:
            sel_idx = inverse_idx
            inverse = True

        lesion: np.ndarray = self.lesions[sel_idx]
        if inverse:
            lesion = lesion.transpose()
        loaded_lesion = torch.Tensor(lesion).unsqueeze(0).unsqueeze(0).cuda()
        with torch.no_grad():
            candidate_lesions = self.vae.sample(x=loaded_lesion, samples=10, explore_dim=13).cpu()
            bbox, crop_lesion = self.find_lesion(candidate_lesions)
            heights = bbox[:,3]-bbox[:,1]
            widths = bbox[:,2]-bbox[:,0]
            areas = heights * widths
            area_idx = find_nearest(areas, area)
            selected_lesion = candidate_lesions[area_idx]
            selected_lesion_crop = crop_lesion[area_idx]
            selected_bbox = bbox[area_idx]
            # plt.figure()
            # plt.imshow(selected_lesion.permute(1,2,0))
            # plt.show()
            # plt.figure()
            # plt.title("Original one")
            # plt.imshow(loaded_lesion[0].permute(1, 2, 0).cpu())
            # plt.show()
            new_a_ratios = (bbox[:,3]-bbox[:,1])/(bbox[:,2]-bbox[:,0])

        min_row = lesion_bbox[1]
        min_col = lesion_bbox[0]
        rows = lesion_bbox[3]
        cols = lesion_bbox[2]
        max_row = min_row + rows
        max_col = min_col + cols
        upscaled_lesion = self.upscale_lesion(
            selected_lesion_crop, selected_bbox
        )
        # input_0 = scaled_down_lesions[sample]
        input_1 = (
            starting_cxr
        )
        result = (1 / 255) * poisson_blend(
            upscaled_lesion, input_1, min_col, max_col, min_row, max_row
        )
        fig, ax = plt.subplots(1,2, figsize=(10,5), dpi=100)
        ax = ax.flatten()
        ax[0].imshow(starting_cxr)

        ax[1].imshow(result)
        ax[1].add_patch(matplotlib.patches.Rectangle((lesion_bbox[0], lesion_bbox[1]), lesion_bbox[2], lesion_bbox[3], facecolor="none", ec='k', lw=1))
        plt.show()
        # results[sample] = torch.Tensor(poisson_edit(addition[sample].permute(1, 2, 0).detach().cpu().numpy(),
        #                                starting_cxr[sample].permute(1, 2, 0).detach().cpu().numpy(),
        #                                mask_lesion[sample].permute(1, 2, 0).detach().cpu().numpy(), (0, 0))).permute(2,0,1)

        # f, axarr = plt.subplots(1,4)

        # plt.imshow(results[0].reshape(results[0].shape[-2:]))
        # draw_rectangle(lesion_bbox.cpu().numpy()[0])
        # plt.show()

        return result

    def predict(self, *, input_image: SimpleITK.Image) -> SimpleITK.Image:
        input_image = SimpleITK.GetArrayFromImage(input_image)
        total_time = time.time()
        if len(input_image.shape) == 2:
            input_image = np.expand_dims(input_image, 0)

        pd_data = pd.read_csv(
            "/opt/algorithm/ct_nodules.csv" if execute_in_docker else "ct_nodules.csv"
        )

        nodule_images = np.zeros(input_image.shape)

        for j in range(len(input_image)):
            t = time.time()
            cxr_img_scaled = input_image[j, :, :]
            nodule_data = [i for i in self.data["boxes"] if i["corners"][0][2] == j]

            for nodule in nodule_data:
                cxr_img_scaled = convert_to_range_0_1(cxr_img_scaled)
                boxes = nodule["corners"]
                # no spacing info in GC with 3D version
                # x_min, y_min, x_max, y_max = boxes[2][0]/spacing_x, boxes[2][1]/spacing_y, boxes[0][0]/spacing_x, boxes[0][1]/spacing_y
                y_min, x_min, y_max, x_max = (
                    boxes[2][0],
                    boxes[2][1],
                    boxes[0][0],
                    boxes[0][1],
                )

                x_min, y_min, x_max, y_max = (
                    int(x_min),
                    int(y_min),
                    int(x_max),
                    int(y_max),
                )

                result = self.place_lesion(None, cxr_img_scaled, (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1))

                # # ------------------------------ Randomly choose ct patch and scale it according to bounding box size.
                # required_diameter = max(x_max - x_min, y_max - y_min)
                # ct_names = pd_data[pd_data["diameter"] > int((required_diameter / 5))][
                #     "img_name"
                # ].values
                # if len(ct_names) < 1:
                #     pd_data[pd_data["diameter"] > int((required_diameter / 10))][
                #         "img_name"
                #     ].values
                #
                # index_ct = random.randint(0, len(ct_names) - 1)
                # path_nodule = (
                #     "/opt/algorithm/nodule_patches/"
                #     if execute_in_docker
                #     else "nodule_patches/"
                # )
                # X_ct_2d_resampled, diameter = process_CT_patches(
                #     os.path.join(path_nodule, ct_names[index_ct]),
                #     os.path.join(path_nodule, ct_names[index_ct].replace("dcm", "seg")),
                #     required_diameter,
                # )
                #
                # crop = cxr_img_scaled[x_min:x_max, y_min:y_max].copy()
                # new_arr = convert_to_range_0_1(X_ct_2d_resampled)
                #
                # # contrast matching:
                # c = contrast_matching(new_arr, cxr_img_scaled[x_min:x_max, y_min:y_max])
                # nodule_contrasted = new_arr * c
                #
                # indexes = nodule_contrasted != np.min(nodule_contrasted)
                # result = poisson_blend(
                #     nodule_contrasted, cxr_img_scaled, y_min, y_max, x_min, x_max
                # )
                # result[x_min:x_max, y_min:y_max] = np.mean(
                #     np.array([crop * 255, result[x_min:x_max, y_min:y_max]]), axis=0
                # )
                # cxr_img_scaled = result.copy()

            nodule_images[j, :, :] = result
        print("total time took ", time.time() - total_time)
        return SimpleITK.GetImageFromArray(nodule_images)


if __name__ == "__main__":
    Nodulegeneration().process()
