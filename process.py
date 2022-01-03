import SimpleITK
import numpy as np
import torch
import matplotlib.pyplot as plt

from evalutils import SegmentationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)
import json
from pathlib import Path
import time

from model_submission.model.inpaint_g import TwostagendGenerator
from model_submission.model.utils import load_network_path
from model_submission.utils.bbox import crop_around_mask_bbox, mask_image
from model_submission.utils.transforms import basic_transform, normalize_cxr


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

        self.crop_size = 256
        self.net = TwostagendGenerator()
        model_save_path = "C:\\Users\\e.marcus\\Models\\chalnode-checkpoints\\ssim551_long_rcnn_finetune\\latest_net_G.pth"
        self.net = load_network_path(self.net, model_save_path, strict=True)
        self.net.eval()

        self.transform = basic_transform()

    def generate_composed_image(self, original_image, masked_image, mask):
        mask_tensor = torch.Tensor(mask)
        original_tensor = self.transform(original_image)
        input_tensor = self.transform(masked_image)
        mask_tensor = torch.unsqueeze(mask_tensor, 0).float()
        input_tensor = torch.unsqueeze(input_tensor, 0).float()
        original_tensor = torch.unsqueeze(original_tensor, 0).float()
        with torch.no_grad():
            output = self.net(input_tensor, mask_tensor)

        composed_image = output[1] * mask + original_tensor * (1 - mask)
        composed_image_np = composed_image.numpy().reshape((self.crop_size, self.crop_size))
        return composed_image_np

    def predict(self, *, input_image: SimpleITK.Image) -> SimpleITK.Image:
        input_image = SimpleITK.GetArrayFromImage(input_image)
        total_time = time.time()
        if len(input_image.shape) == 2:
            input_image = np.expand_dims(input_image, 0)
        
        nodule_images = np.zeros(input_image.shape)
        
        for j in range(len(input_image)):
            t = time.time()
            cxr_img_scaled = input_image[j, :, :]
            cxr_img_scaled = normalize_cxr(cxr_img_scaled)
            nodule_data = [i for i in self.data['boxes'] if i['corners'][0][2] == j]

            # loop over different images
            for nodule in nodule_data:
                boxes = nodule['corners']
                y_min, x_min, y_max, x_max = boxes[2][0], boxes[2][1], boxes[0][0], boxes[0][1]
                mask_bbox = [int(x_min), int(y_min), int(x_max) - int(x_min), int(y_max) - int(y_min)]

                cropped_cxr, new_mask_bbox, crop_bbox = crop_around_mask_bbox(cxr_img_scaled, mask_bbox)
                cropped_masked_cxr, cropped_mask = mask_image(cropped_cxr, new_mask_bbox)
                composed_img = self.generate_composed_image(cropped_cxr, cropped_masked_cxr, cropped_mask)

                c_x1, c_y1, c_w, c_h = crop_bbox
                cxr_img_scaled[c_y1: c_y1+c_h, c_x1: c_x1+c_w] = (composed_img + 1) / 2  # undo normalization on output

            print("time for image: ", time.time()-t)
            result = cxr_img_scaled.copy()
            result *= 4095  # TODO: undo normalization here correct?
            nodule_images[j,:,:] = result
            plt.imshow(result)
            plt.show()
        print('total time took ', time.time()-total_time)
        return SimpleITK.GetImageFromArray(nodule_images)

if __name__ == "__main__":
    Nodulegeneration().process()

                                                                 
                                                                 
