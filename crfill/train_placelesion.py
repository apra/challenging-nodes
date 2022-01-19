import pdb
import sys
import torch
import numpy as np
from collections import OrderedDict
from options.train_options import TrainOptions
import data
from util.iter_counter import IterationCounter
from logger import Logger
from torchvision.utils import make_grid
from trainers import create_trainer
from torch.utils import tensorboard

# from torch.utils.tensorboard import SummaryWriter
from util.util import set_all_seeds
from trainers.pix2pixplacelesion_trainer import Pix2PixPlaceLesionTrainer
from trainers.vae_trainer import BaselineVAEOpts
from models.vae_model import vaemodel

from torchvision import datasets, transforms
from util.plot_util import draw_bounding_boxes

# parse options
# temporary fix for LISA
# torch.set_num_threads(24)
opt = TrainOptions().parse()

set_all_seeds(opt.seed)

# print options to help debugging
print(" ".join(sys.argv))

# load the dataset
dataloader_train, dataloader_val = data.create_dataloader_trainval(opt)

# create trainer for our model
# trainer = create_trainer(opt)

vaeopts = BaselineVAEOpts.copy()
vaeopts["trainer"]["optimizer_config"]["lr"] = opt.lr
vaeopts["model"]["latent_size"] = opt.latent_size
vaeopts["model"]["sigma"] = opt.sigma
vaeopts["model"]["beta_kl"] = opt.beta_kl
vaeopts["model"]["encoder_params"]["downsample"] = opt.downsample

vaeModel = vaemodel(opt=opt, **vaeopts["model"])
vaeModel.cuda()
vaeModel.load()

model = vaeModel

# create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader_train))

# create tool for visualization
writer = Logger(f"output/{opt.name}")

ts_writer = tensorboard.SummaryWriter(f"{opt.checkpoints_dir}/tensorboard")


# torch.multiprocessing.set_sharing_strategy('file_system')

for i, data_i in enumerate(dataloader_train, start=iter_counter.epoch_iter):
    iter_counter.record_one_iteration()

print("Training was successfully finished.")
