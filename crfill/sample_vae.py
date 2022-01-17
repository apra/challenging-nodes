import pdb
import sys
import torch
import numpy as np
from collections import OrderedDict
from options.train_options import TrainOptions, SampleOption
import data
from util.iter_counter import IterationCounter
from logger import Logger
from torchvision.utils import make_grid
from trainers import create_trainer
from torch.utils import tensorboard

# from torch.utils.tensorboard import SummaryWriter
from util.util import set_all_seeds
from trainers.vae_trainer import VAETrainer
from pathlib import Path
from torchvision import datasets, transforms
from util.plot_util import draw_bounding_boxes

# parse options
# temporary fix for LISA
# torch.set_num_threads(24)
opt = SampleOption().parse()

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

output_dir = Path(opt.checkpoints_dir)
output_dir.mkdir(parents=True, exist_ok=True)
with torch.no_grad():
    model.eval()
    for i, data in enumerate(dataloader_train):
        model.sample(data["inputs"], out_dir=output_dir/Path("from_input_sample"))
        model.traversal(out_dir=output_dir, x=data['inputs'])
        # ts_writer.add_image("sampling/results", make_grid((result), nrow=4), i)
        # ts_writer.add_image("sampling/inputs", make_grid((data["inputs"])), i)

        model.sample(out_dir=output_dir/Path("random_sample"), samples=8)
        break
        #ts_writer.add_image("sampling/random", make_grid(result, nrow=4), i)
