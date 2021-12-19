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
model = trainer.model

# create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader_train))

# create tool for visualization
writer = Logger(f"output/{opt.name}")

ts_writer = tensorboard.SummaryWriter(f"{opt.checkpoints_dir}/tensorboard")

trainer.save("latest")

# torch.multiprocessing.set_sharing_strategy('file_system')

for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)
    for i, data_i in enumerate(dataloader_train, start=iter_counter.epoch_iter):
        iter_counter.record_one_iteration()
        # train discriminator
        if i % opt.D_steps_per_G == 0:
            trainer.run_generator_one_step(data_i, i)

        if iter_counter.needs_displaying():
            losses = trainer.get_latest_losses()
            for k, v in losses.items():
                ts_writer.add_scalar(
                    f"train/{k}", v.mean().item(), iter_counter.total_steps_so_far
                )
            writer.write_console(
                epoch, iter_counter.epoch_iter, iter_counter.time_per_iter
            )
            input_name = "inputs"
            num_print = min(4, data_i[input_name].size(0))
            # writer.add_single_image('inputs',
            #         (make_grid(data_i['full_image'][:num_print])+1)/2,
            #         iter_counter.total_steps_so_far)
            # ts_writer.add_image('inputs',
            #                 draw_bounding_boxes(data_i['full_image'],data_i['bounding_box']),
            #                 iter_counter.total_steps_so_far)
            ts_writer.add_image(
                "train/original_cropped",
                make_grid((data_i[input_name][:num_print])),
                iter_counter.total_steps_so_far,
            )

            out = trainer.model.forward(data_i["inputs"])
            infer_out = out["reconstructed"]
            vis = make_grid(data_i["inputs"][:num_print])
            ts_writer.add_image("train/infer_in", vis, iter_counter.total_steps_so_far)
            vis = make_grid(infer_out[:num_print])
            vis = torch.clamp(vis, 0, 1)
            ts_writer.add_image("train/infer_out", vis, iter_counter.total_steps_so_far)
        if iter_counter.needs_validation():
            print(
                "saving the latest model (epoch %d, total_steps %d)"
                % (epoch, iter_counter.total_steps_so_far)
            )
            trainer.save("epoch%d_step%d" % (epoch, iter_counter.total_steps_so_far))
            trainer.save("latest")
            iter_counter.record_current_iter()

            print("doing validation")
            model.eval()
            num = 0
            psnr_total = 0
            for ii, data_ii in enumerate(dataloader_val):
                with torch.no_grad():
                    generated = model(data_ii["inputs"])
                    generated = generated["reconstructed"].cpu()
                generated = (generated) * 255
                gt = data_ii[input_name]
                bsize, c, h, w = gt.shape
                gt = (gt + 1) / 2 * 255
                mse = ((generated - gt) ** 2).sum(3).sum(2).sum(1)
                mse /= c * h * w
                psnr = 10 * torch.log10(255.0 * 255.0 / (mse + 1e-8))
                psnr_total += psnr.sum().item()
                num += bsize
            psnr_total /= num
            ts_writer.add_scalar(
                "val/psnr", psnr_total, iter_counter.total_steps_so_far
            )
            model.train()
    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()
    trainer.save("latest")

print("Training was successfully finished.")
