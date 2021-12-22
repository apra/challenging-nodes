import torch
from models.networks.sync_batchnorm import DataParallelWithCallback
from models.vae_model import vaemodel

# from models.pix2pix_model import Pix2PixModel

BaselineVAEOpts = {
    "batch_size": 64,
    "trainer": {
        "steps": 500_000,
        "use_exp_decay": True,
        "exp_decay_rate": 0.5,
        "exp_decay_steps": 100_000,
        "optimizer_config": {"alg": "Adam", "lr": 0.0005},
    },
    "model": {
        "height": 32,
        "width": 32,
        "num_slots": 1,
        "sigma": 0.06,
        "beta_kl": 1,
        "latent_size": 64,
        "encoder_params": {
            "layers_per_block_per_layer": [8, 8, 8, 8, 8, 8, 8, 8],
            "channel_size_per_layer": [64, 64, 128, 128, 128, 128, 256, 256],
            "num_layers_per_resolution": [
                2,
                2,
                2,
                2,
            ],  # same logic as in the decoder but with avg pool 2d
            "downsample": 4,
            "mlp_hidden_size": 1024,
        },
        "decoder_params": {
            "layers_per_block_per_layer": [8, 8, 8, 8, 8, 8, 8, 8],
            "broadcast_size": 8,  # ignored with type :: mlp
            "channel_size_per_layer": [256, 256, 128, 128, 128, 128, 64, 64],
            "num_layers_per_resolution": [2, 2, 2, 2],
            "mlp_hidden_size": 1024,
            "architecture_type": "mlp",
        },
    },
}


class VAETrainer:
    def __init__(self, opt):
        self.opt = opt
        BaselineVAEOpts["trainer"]["optimizer_config"]["lr"] = self.opt.lr
        BaselineVAEOpts["model"]["latent_size"] = self.opt.latent_size
        BaselineVAEOpts["model"]["sigma"] = self.opt.sigma
        BaselineVAEOpts["model"]["beta_kl"] = self.opt.beta_kl
        BaselineVAEOpts["model"]["encoder_params"]['downsample'] = self.opt.downsample

        self.model = vaemodel(opt=opt, **BaselineVAEOpts["model"])
        if len(opt.gpu_ids) > 0:
            self.model = DataParallelWithCallback(self.model, device_ids=opt.gpu_ids)
            self.pix2pix_model_on_one_gpu = self.model.module
        else:
            self.pix2pix_model_on_one_gpu = self.model

        self.generated = None
        self.inputs = None
        self.mean_path_length = torch.Tensor([0])
        self.lr = BaselineVAEOpts["trainer"]["optimizer_config"]["lr"]
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.old_lr = self.lr

    def run_generator_one_step(self, data, i):
        self.optimizer.zero_grad()
        data["inputs"] = data["inputs"].to("cuda")
        out = self.model(data["inputs"])
        self.losses = {
            "loss": out["loss"],
            "neg_log_p_x": out["neg_log_p_x"],
            "kl_latent": out["kl_latent"],
            "mse": out["mse"],
        }
        loss = out["loss"]
        loss.backward()
        self.optimizer.step()
        self.generated = out["reconstructed"]
        self.inputs = data

    def run_discriminator_one_step(self, data, i):
        pass

    def get_latest_losses(self):
        return self.losses

    def get_latest_generated(self):
        return self.generated

    def get_latest_inputs(self):
        return self.inputs

    def update_learning_rate(self, epoch):
        self.update_learning_rate(epoch)

    def save(self, epoch):
        self.pix2pix_model_on_one_gpu.save(epoch)

    ##################################################################
    # Helper functions
    ##################################################################

    def update_learning_rate(self, epoch):
        if epoch > self.opt.niter:
            lrd = self.lr / self.opt.niter_decay
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            print(f"Learning rate changed to {new_lr}")
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = new_lr

            self.old_lr = new_lr
