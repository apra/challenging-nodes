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
from trainers.vae_trainer import VAETrainer

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
trainer = VAETrainer(opt)
model = trainer.model_on_one_gpu

model.load()

ts_writer = tensorboard.SummaryWriter(f"{opt.checkpoints_dir}/tensorboard")

for i, data in enumerate(dataloader_train):
    result = model.sample(data["inputs"])
    ts_writer.add_image("sampling/results", make_grid((result), nrow=4), i)
    ts_writer.add_image("sampling/inputs", make_grid((data["inputs"])), i)

    result = model.sample()
    ts_writer.add_image("sampling/random", make_grid(result, nrow=4), i)
