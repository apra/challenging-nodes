import torch

from models import networks
from data.custom_train_negative_dataset import CustomTrainNegativeDataset
from data.custom_train_dataset import CustomTrainDataset
from torch.utils.data import DataLoader
from models.networks.inpaint_g import *
from util.metadata_utils import get_paths_negatives, get_paths_and_nodules
import matplotlib.pyplot as plt


class option():
    def __init__(self):
        #self.fastercnn_loc = "models/fasterrcnn.pth"
        self.fastercnn_loc = "C:\\Users\\e.marcus\\PycharmProjects\\challenging-nodes\\crfill\\models\\fasterrcnn.pth"
        self.gpu_ids = [0]
        self.crop_around_mask_size = 256
        #self.train_nodule_list = 'C:\\Users\\e.marcus\\PycharmProjects\\challenging-nodes\\datasets\\node21\\cxr_images\\proccessed_data\\metadata.csv'
        self.preprocess_mode = 'none'
        self.isTrain = True
        self.no_flip = True
        self.use_cam = True
        self.norm_type = 1
        self.use_th = False
        self.th = 0.5
        self.no_mk = False
        self.baseG = 'baseconv'
        #self.paths_and_nodules = 'C:\\Users\\e.marcus\\PycharmProjects\\challenging-nodes\\datasets\\node21\\'
        self.mod = 'train'
        self.include_chexpert = False
        self.include_mimic = False
        self.node21_resample_count = 0
        self.seed = 1
        self.num_folds = 5
        self.fold = 1
        self.load_size = 1024
        self.k_means_clusters = 3
        #self.train_image_dir = '/mnt'
        self.train_image_dir = 'C:\\Users\\e.marcus\\PycharmProjects\\challenging-nodes\\datasets\\'


opt = option()
paths_negative = get_paths_negatives(opt.train_image_dir)
paths_positive = get_paths_and_nodules(opt.train_image_dir, opt.include_chexpert,
                                       opt.include_mimic, opt.node21_resample_count)
cdt = CustomTrainNegativeDataset()
cdt.initialize(option(), paths_negative, opt.mod)

cdtpos = CustomTrainDataset()
cdtpos.initialize(option(), paths_positive, opt.mod)

dataloader = DataLoader(
        cdt,
        batch_size=1
    )

dataloaderpos = DataLoader(
        cdtpos,
        batch_size=1
    )

it = iter(dataloader)
itpos = iter(dataloaderpos)



print(torch.cuda.device_count())
activation = {}

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook




rcnn_mod = networks.define_DRCNN(opt)
hook = rcnn_mod.fastercnn_model.backbone.body.layer4.register_forward_hook(get_activation('layer4'))
rcnn_mod.eval()
targets = [{'boxes': torch.unsqueeze(torch.LongTensor([1,2,3,4]).cuda(), 0).cuda(),
            'labels': torch.LongTensor([1]).cuda()}]
img = it.next()
imgpos = itpos.next()
outp = rcnn_mod(img['inputs'].cuda(), targets)
print(outp)
print(torch.mean(activation['layer4']))

outp2 = rcnn_mod(imgpos['real_image'].cuda(), targets)
print(outp2)
print(torch.mean(activation['layer4']))
f, axarr = plt.subplots(1, 2)
axarr[0].imshow((img['inputs'].detach().numpy().reshape((256, 256))), cmap='gray')
axarr[1].imshow(imgpos['real_image'].detach().numpy().reshape((256, 256)), cmap='gray')
plt.show()


