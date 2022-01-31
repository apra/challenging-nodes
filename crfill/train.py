import sys

import torch
from torch.utils import tensorboard
from torchvision.utils import make_grid

import data
from logger import Logger
from options.train_options import TrainOptions
from trainers.pix2pix_trainer import Pix2PixTrainer
from util.iter_counter import IterationCounter
# from torch.utils.tensorboard import SummaryWriter
from util.util import set_all_seeds

# parse options
opt = TrainOptions().parse()

set_all_seeds(opt.seed)

# load the dataset
dataloader_train, dataloader_val = data.create_dataloader_trainval(opt)

# create trainer for our model
trainer = Pix2PixTrainer(opt)
model = trainer.model

# create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader_train))

# create tool for visualization
writer = Logger(f"output/{opt.name}")

ts_writer = tensorboard.SummaryWriter(f'{opt.checkpoints_dir}/tensorboard')

trainer.save('latest')

for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)
    for i, data_i in enumerate(dataloader_train, start=iter_counter.epoch_iter):
        iter_counter.record_one_iteration()

        # train discriminator
        if not opt.freeze_D:
            trainer.run_discriminator_one_step(data_i, i)

        # Training
        # train generator
        if i % opt.D_steps_per_G == 0:
            trainer.run_generator_one_step(data_i, i)

        if iter_counter.needs_displaying():
            losses = trainer.get_latest_losses()
            for k, v in losses.items():
                ts_writer.add_scalar(f"train/{k}", v.mean().item(), iter_counter.total_steps_so_far)
            writer.write_console(epoch, iter_counter.epoch_iter, iter_counter.time_per_iter)
            input_name = 'real_image'
            num_print = min(4, data_i[input_name].size(0))
            ts_writer.add_image('train/original_cropped',
                               make_grid((data_i[input_name][:num_print]+1)/2),
                               iter_counter.total_steps_so_far)

            infer_out,inp = trainer.model.forward(data_i, mode='inference')
            vis = (make_grid(inp[:num_print])+1)/2
            ts_writer.add_image('train/infer_in',
                                    vis,
                                    iter_counter.total_steps_so_far)
            vis = (make_grid(infer_out[:num_print])+1)/2
            vis = torch.clamp(vis, 0,1)
            ts_writer.add_image('train/infer_out',
                    vis,
                    iter_counter.total_steps_so_far)
            generated = trainer.get_latest_generated()
            for k,v in generated.items():
                if v is None:
                    continue
                if 'label' in k:
                    vis = make_grid(v[:num_print].expand(-1,3,-1,-1))[0]
                    writer.add_single_label(k,
                            vis,
                            iter_counter.total_steps_so_far)
                else:
                    if v.size(1) == 3:
                        vis = (make_grid(v[:num_print])+1)/2
                        vis = torch.clamp(vis, 0,1)
                    else:
                        vis = make_grid(v[:num_print])
                    writer.add_single_image(k,
                            vis,
                            iter_counter.total_steps_so_far)
        if iter_counter.needs_validation():
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, iter_counter.total_steps_so_far))
            trainer.save('epoch%d_step%d'%
                    (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            iter_counter.record_current_iter()

    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()
    trainer.save('latest')

print('Training was successfully finished.')
