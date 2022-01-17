import SimpleITK
import numpy as np
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from evalutils import SegmentationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)
import json
from pathlib import Path
import time
from copy import deepcopy

from model_submission.model.inpaint_g import TwostagendGenerator
from model_submission.model.utils import load_network_path
from model_submission.utils.bbox import crop_around_mask_bbox, mask_image, intersection_over_union
from model_submission.utils.transforms import basic_transform, normalize_cxr

#import matplotlib.pyplot as plt
# This parameter adapts the paths between local execution and execution in docker. You can use this flag to switch between these two modes.
# For building your docker, set this parameter to True. If False, it will run process.py locally for test purposes.
execute_in_docker = True


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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net1 = TwostagendGenerator()
        self.net2 = TwostagendGenerator()
        self.judge = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
        num_classes = 2  # 1 class (nodule) + background
        in_features = self.judge.roi_heads.box_predictor.cls_score.in_features
        self.judge.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        model_save_path1 = "model_submission/model/modelr25.pth"
        model_save_path2 = "model_submission/model/modelr.pth"
        model_save_path3 = "model_submission/model/fasterrcnn.pth"
        self.net1 = load_network_path(self.net1, model_save_path1, strict=True)
        self.net1.to(self.device)
        self.net2 = load_network_path(self.net2, model_save_path2, strict=True)
        self.net2.to(self.device)

        self.judge.load_state_dict(
            torch.load(
                model_save_path3,
                map_location=self.device,
            )
        )
        self.judge.to(self.device)

        self.net1.eval()
        self.net2.eval()
        self.judge.eval()

        self.transform = basic_transform()

    def generate_composed_image(self, original_image, masked_image, mask):
        mask_tensor = torch.Tensor(mask)
        original_tensor = self.transform(original_image)
        input_tensor = self.transform(masked_image)

        mask_tensor = torch.unsqueeze(mask_tensor, 0).float().to(self.device)
        input_tensor = torch.unsqueeze(input_tensor, 0).float().to(self.device)
        original_tensor = torch.unsqueeze(original_tensor, 0).float().to(self.device)
        with torch.no_grad():
            output1 = self.net1(input_tensor, mask_tensor)
            output2 = self.net2(input_tensor, mask_tensor)

        composed_output = (output1[1] + output2[1]) / 2
        composed_sum = composed_output * mask_tensor + original_tensor * (1 - mask_tensor)
        composed_1 = output1[1] * mask_tensor + original_tensor * (1 - mask_tensor)
        composed_2 = output2[1] * mask_tensor + original_tensor * (1 - mask_tensor)
        # composed_sum_np = composed_sum.cpu().numpy().reshape((self.crop_size, self.crop_size))
        # composed_1_np = composed_1.cpu().numpy().reshape((self.crop_size, self.crop_size))
        # composed_2_np = composed_2.cpu().numpy().reshape((self.crop_size, self.crop_size))
        # f, axarr = plt.subplots(1, 3)
        # axarr[0].imshow(composed_sum_np, cmap='gray')
        # axarr[1].imshow(composed_1_np, cmap='gray')
        # axarr[2].imshow(composed_2_np, cmap='gray')
        # plt.show()
        return [composed_1, composed_2, composed_sum]

    def get_max_iou(self, prediction, target_box):
        target_box = target_box.to('cpu').numpy()[0]
        iou_list = []
        for box in prediction[0]['boxes']:
            box = box.detach().to('cpu').numpy()
            iou_list.append(intersection_over_union(target_box, box))

        scores = prediction[0]['scores']
        max_iou = max(iou_list)
        max_box_idx = np.argmax(iou_list)
        max_score = scores[max_box_idx]
        return max_iou, max_score

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
            result = deepcopy(cxr_img_scaled)
            # plt.imshow(cxr_img_scaled, cmap='gray')
            # plt.show()
            nodule_data = [i for i in self.data['boxes'] if i['corners'][0][2] == j]

            # loop over different images
            for nodule in nodule_data:
                #print('new nodule ---')
                boxes = nodule['corners']
                x_min, y_min, x_max, y_max = boxes[2][0], boxes[2][1], boxes[0][0], boxes[0][1]
                mask_bbox = [int(x_min), int(y_min), int(x_max) - int(x_min), int(y_max) - int(y_min)]

                b_bbox, best_im = None, None
                best_score = 0
                for i in range(5):
                    #print("\n new view")
                    cropped_cxr, new_mask_bbox, crop_bbox = crop_around_mask_bbox(cxr_img_scaled, mask_bbox)
                    cropped_masked_cxr, cropped_mask = mask_image(cropped_cxr, new_mask_bbox)
                    composed_imgages = self.generate_composed_image(cropped_cxr, cropped_masked_cxr, cropped_mask)
                    #plt.imshow(cropped_cxr, cmap='gray')
                    #plt.show()
                    c_x1, c_y1, c_w, c_h = crop_bbox
                    c_x2, c_y2 = c_x1 + c_w, c_y1 + c_h

                    #f, axarr = plt.subplots(1, 3)
                    for idx, image in enumerate(composed_imgages):

                        cxr_tensor = torch.cuda.FloatTensor(cxr_img_scaled)
                        cxr_tensor = cxr_tensor.view((1, 1, 1024, 1024))
                        cxr_tensor[:, :, c_y1: c_y1+c_h, c_x1: c_x1+c_w] = (image + 1) / 2  # undo normalization on output
                        if torch.cuda.is_available():
                            targets = [{'boxes': torch.unsqueeze(torch.Tensor([x_min, y_min, x_max, y_max]).cuda(), 0),
                                        'labels': torch.LongTensor([1]).cuda()}]  # label=1 for tumor class
                        else:
                            targets = [{'boxes': torch.unsqueeze(torch.Tensor([x_min, y_min, x_max, y_max]), 0),
                                        'labels': torch.LongTensor([1])}]  # label=1 for tumor class
                        #self.judge.eval()
                        prediction = self.judge(cxr_tensor)
                        #self.judge.train()
                        #print(prediction)
                        try:
                            max_iou, max_score = self.get_max_iou(prediction, targets[0]['boxes'])
                            #print(f'Max iou: {max_iou:.2f} and max score {max_score:.2f}')
                            if (max_score+max_iou)/2 > best_score:
                                #print(f'updated best with {(max_score+max_iou)/2}')
                                b_bbox = [c_x1, c_y1, c_x2, c_y2]
                                best_im = image
                                #best_full = cxr_tensor
                                best_score = max_score
                        except:
                            #print('excepted')
                            if best_im is None:
                                #print('nonned')
                                b_bbox = [c_x1, c_y1, c_x2, c_y2]
                                best_im = image
                        del prediction
                        #axarr[idx].imshow(image.cpu().numpy().reshape((self.crop_size, self.crop_size)), cmap='gray')
                        #axarr[idx].title.set_text(f'Max iou: {max_iou:.2f} and max score {max_score:.2f}')

                        #plt.imshow(image.detach().cpu().numpy().reshape((256, 256)), cmap='gray')
                        #plt.show()
                        # loss_dict = self.judge(cxr_tensor, targets)
                        # losses = []
                        # for value in loss_dict.values():
                        #     val = value.detach().cpu().numpy()
                        #     losses.append(val)
                        # print(f"avg loss: {sum(losses)/4:.2f} and total {sum(losses):.2f}")
                        # axarr[idx].title.set_text(f'avg: {sum(losses)/4:.2f} and tot {sum(losses):.2f}')
                        # print(loss_dict)

                #plt.imshow(best_im.detach().cpu().numpy().reshape((256, 256)), cmap='gray')
                #plt.show()
                # plt.imshow(best_full.detach().cpu().numpy().reshape((1024, 1024)), cmap='gray')
                # plt.show()
                try:
                    replace_crop = best_im.detach().cpu().numpy().reshape((self.crop_size, self.crop_size))
                    result[b_bbox[1]: b_bbox[3], b_bbox[0]: b_bbox[2]] = (replace_crop + 1) / 2
                    #del b_bbox, best_im, best_score  # make sure they cant be reused next round
                except:
                    #print('no best found, continuing')
                    replace_crop = image.detach().cpu().numpy().reshape((self.crop_size, self.crop_size))
                    result[c_y1: c_y1+c_h, c_x1: c_x1+c_w] = (replace_crop + 1) / 2
            #result = cxr_tensor.detach().cpu().numpy().reshape((1024, 1024))
            print("time for image: ", time.time()-t)
            result *= 255  # same normalization they did as in the baseline
            nodule_images[j, :, :] = result
        print('total time took ', time.time()-total_time)
        return SimpleITK.GetImageFromArray(nodule_images)


if __name__ == "__main__":
    Nodulegeneration().process()

                                                                 
                                                                 
