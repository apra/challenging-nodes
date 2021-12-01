import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
import torchvision.models.detection
from torch.nn.functional import normalize
from models.networks.base_network import BaseNetwork
from models.networks.utils import dis_conv
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class DeepFillDiscriminator(BaseNetwork):
    def __init__(self, opt):
        super(DeepFillDiscriminator, self).__init__()
        cnum = 64
        self.img_channels = 1
        self.conv1 = nn.utils.spectral_norm(dis_conv(self.img_channels+1, cnum))
        self.conv2 = nn.utils.spectral_norm(dis_conv(cnum, cnum*2))
        self.conv3 = nn.utils.spectral_norm(dis_conv(cnum*2, cnum*4))
        self.conv4 = nn.utils.spectral_norm(dis_conv(cnum*4, cnum*4))
        self.conv5 = nn.utils.spectral_norm(dis_conv(cnum*4, cnum*4))
        self.conv6 = nn.utils.spectral_norm(dis_conv(cnum*4, cnum*4))


    def forward(self, x, mask=None):
        bsize, ch, height, width = x.shape
        if mask is None:
            ones_x = torch.ones(bsize, 1, height, width).to(x.device)
        else:
            ones_x = mask
        x = torch.cat([x, ones_x], 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x


class FasterRCNNDiscriminator(BaseNetwork):
    def __init__(self, opt):
        super(FasterRCNNDiscriminator, self).__init__()
        self.fastercnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
        num_classes = 2
        in_features = self.fastercnn_model.roi_heads.box_predictor.cls_score.in_features
        self.fastercnn_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def forward(self, x):
        # TODO: get the correct bbox here, such that it can forward correctly through the model
        return x


if __name__ == "__main__":
    pass
