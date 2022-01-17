import pdb
import warnings

import torch
import models.networks as networks
import util.util as util
from models.create_mask import MaskCreator
import random
import numpy as np
from models.vae_model import vaemodel

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
        "height": 64,
        "width": 64,
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


def conv3x3(in_channels, out_channels, stride=1):
    return torch.nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )


# Residual block
class ResidualBlock(torch.nn.Module):
    def __init__(self, ch, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(ch, ch, stride)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = conv3x3(ch, ch)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        # out = self.relu(out)
        out = self.conv2(out)
        out += residual
        # out = self.relu(out)
        return out


class placelesionmodel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        parser.add_argument(
            "--path_objectshape_base", type=str, default="", help="path obj base"
        )
        parser.add_argument(
            "--path_objectshape_list", type=str, default="", help="path obj list"
        )
        parser.add_argument(
            "--update_part", type=str, default="all", help="update part"
        )
        parser.add_argument(
            "--d_mask_in", action="store_true", help="if specified, d mask in"
        )
        parser.add_argument(
            "--no_fine_loss",
            action="store_true",
            help="if specified, do *not* use refinementstageloss",
        )
        parser.add_argument(
            "--load_pretrained_g", type=str, required=False, help="load pt g"
        )
        parser.add_argument(
            "--load_pretrained_d", type=str, required=False, help="load pt d"
        )
        parser.add_argument(
            "--fastercnn_loc",
            default="models/fasterrcnn.pth",
            help="location of the fasterrcnn.pth",
        )
        parser.add_argument("--no_sharp_skip_connection", action="store_true")
        parser.add_argument("--generator_weight", type=float, default=1)
        parser.add_argument("--discriminator_weight", type=float, default=1)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.FloatTensor = (
            torch.cuda.FloatTensor if self.use_gpu() else torch.FloatTensor
        )
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() else torch.ByteTensor

        (
            self.netG,
            self.netD,
            self.netDRCNN,
            self.place_lesion,
        ) = self.initialize_networks(opt)

        self.netG.load()
        if opt.isTrain and opt.load_pretrained_d is not None:
            print(f"looad {opt.load_pretrained_d}")
            self.netD = util.load_network_path(self.netD, opt.load_pretrained_d)

        # set loss functions
        if opt.isTrain:
            # self.mask_creator = MaskCreator(opt.path_objectshape_list, opt.path_objectshape_base)
            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt
            )
            self.criterionFeat = torch.nn.L1Loss()
            if opt.vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)

    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.
    def forward(self, data, mode):
        negative, positive, mask, crop_bbox, lesion_bbox, cxr = self.preprocess_input(
            data
        )

        if mode == "generator":
            g_loss, coarse_image, composed_image = self.compute_generator_loss(
                negative, positive, mask, cxr
            )
            generated = {"coarse": coarse_image, "composed": composed_image}
            return g_loss, negative, generated
        elif mode == "discriminator":
            d_loss = self.compute_discriminator_loss(
                negative, positive, mask, lesion_bbox
            )
            return d_loss, data["inputs"]
        elif mode == "inference":
            with torch.no_grad():
                lesion = self.generate_fake(negative, mask)
                composed_image = self.place_addition_on_cxr(lesion, negative, mask)
            return composed_image, negative
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        G_params = self.netG.get_param_list(opt.update_part) + [
            p for name, p in self.place_lesion.named_parameters()
        ]
        # G_params = [p for name, p in self.netG.named_parameters() \
        #        if (not name.startswith("coarse"))]
        if opt.isTrain:
            D_params = list(self.netD.parameters())

        beta1, beta2 = opt.beta1, opt.beta2
        if opt.no_TTUR:
            G_lr, D_lr = opt.lr, opt.lr
        else:
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        return optimizer_G, optimizer_D

    def save(self, epoch):
        util.save_network(self.netG, "G", epoch, self.opt)
        util.save_network(self.netD, "D", epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):
        vaeopts = BaselineVAEOpts.copy()
        vaeopts["trainer"]["optimizer_config"]["lr"] = self.opt.lr
        vaeopts["model"]["latent_size"] = self.opt.latent_size
        vaeopts["model"]["sigma"] = self.opt.sigma
        vaeopts["model"]["beta_kl"] = self.opt.beta_kl
        vaeopts["model"]["encoder_params"]["downsample"] = self.opt.downsample

        vaeModel = vaemodel(opt=opt, **vaeopts["model"])
        vaeModel.cuda()
        if opt.isTrain:
            netD = networks.define_D(opt)
            netDRCNN = networks.define_DRCNN(opt)
        else:
            netD = None
            netDRCNN = None

        if not opt.isTrain or opt.continue_train:
            if opt.isTrain:
                netD = util.load_network(netD, "D", opt.which_epoch, opt)
                raise NotImplementedError(
                    "Loading DRCNN for continuing training not implemented yet"
                )
        place_lesion = torch.nn.Sequential(
            torch.nn.Conv2d(1, 3, 3, 1, 1),
            ResidualBlock(3),
            torch.nn.Conv2d(3, 1, 3, 1, 1),
        )
        return vaeModel, netD, netDRCNN, place_lesion

    # preprocess the input, such as moving the tensors to GPUs and
    # transforming the label map to one-hot encoding
    # |data|: dictionary of the input data

    def preprocess_input(self, data):
        if (
            self.use_gpu()
        ):  # this command is irrelevant, all data will already be at the GPU due to DataParallel in pix2pixtrainer
            data["neg_cropped_normalized_cxr"] = data[
                "neg_cropped_normalized_cxr"
            ].cuda()
            data["pos_cropped_normalized_cxr"] = data[
                "pos_cropped_normalized_cxr"
            ].cuda()
            data["neg_cropped_mask"] = data["neg_cropped_mask"].cuda()

        return (
            data["neg_cropped_normalized_cxr"],
            data["pos_cropped_normalized_cxr"],
            data["neg_cropped_mask"],
            data["neg_crop_bbox"],
            data["neg_lesion_bbox"],
            data["neg_cxr"],
        )

    def g_image_loss(
        self,
        coarse_image,
        negative,
        composed_image,
        positive,
        mask,
        composed_full_image,
        full_image_bbox,
    ):
        G_losses = {}

        # Faster RCNN loss -- For aux loss, composed_full_image and full_image_bbox will be passed as None
        # if composed_full_image is not None:
        #     rcnn_losses = self.RCNN_loss(composed_full_image, full_image_bbox)
        #     G_losses["RCNN"] = rcnn_losses
        #     # G_losses['RCNN'] = self.criterionGAN(pred_rcnn, True, for_discriminator=False)

        # 'Normal' Adversarial
        if not self.opt.no_gan_loss and not self.opt.no_fine_loss:
            pred_fake, pred_real = self.discriminate(composed_image, positive, mask)

            G_losses["GAN"] = (
                self.criterionGAN(pred_fake, True, for_discriminator=False)
                * self.opt.generator_weight
            )

        if self.opt.vgg_loss and not self.opt.no_fine_loss:
            G_losses["VGG"] = (
                self.criterionVGG(negative, positive) * self.opt.lambda_vgg
            )
        if not self.opt.no_l1_loss:
            if coarse_image is not None:
                G_losses["L1_coarse"] = (
                    torch.nn.functional.l1_loss(coarse_image, negative)
                    * self.opt.beta_l1
                )
            if not self.opt.no_fine_loss:
                G_losses["L1_fine"] = (
                    torch.nn.functional.l1_loss(composed_image, negative)
                    * self.opt.beta_l1
                )
        return G_losses

    def place_addition_on_cxr(self, addition, starting_cxr, lesion_bbox):
        max_attenuation = 0.25
        final_addition = addition.clone()
        new_addition = torch.empty_like(addition)
        threshold = 1 / 4096
        starting_cxr = (starting_cxr + 1) / 2
        starting_cxr[starting_cxr < threshold] = threshold
        final_addition[final_addition < threshold] = threshold
        addition = final_addition
        for i, (star, add) in enumerate(zip(starting_cxr, addition)):
            new_addition[i] = max_attenuation * add / add.max() + 1 - max_attenuation

        addition = new_addition
        if (
            addition.shape[2] < starting_cxr.shape[2]
            or addition.shape[3] < starting_cxr.shape[3]
        ):

            final_addition = torch.zeros_like(starting_cxr)
            for sample in range(final_addition.shape[0]):
                x = lesion_bbox[sample][0].item()
                y = lesion_bbox[sample][1].item()
                w = lesion_bbox[sample][2].item()
                h = lesion_bbox[sample][3].item()
                width = addition[sample].shape[1]
                height = addition[sample].shape[2]

                if width > w - 1:
                    delta = width - w
                    rnd_x = int(max(x - delta, 0))
                else:
                    max_x = max(x + 1, x + w - width)
                    rnd_x = np.random.randint(x, max_x)
                if height > h - 1:
                    delta = height - h
                    rnd_y = int(max(y - delta, 0))
                else:
                    max_y = max(y + 1, y + h - height)
                    rnd_y = np.random.randint(y, max_y)

                final_addition[sample] = addition[sample].min()
                final_addition[
                    sample, :, rnd_x : rnd_x + width, rnd_y : rnd_y + height
                ] = addition[sample]
        addition = final_addition
        return self.place_lesion(((starting_cxr * addition) - 0.5) / 0.5)

    def compute_generator_loss(
        self, negative, positive, mask, crop_bbox, lesion_bbox, cxr, cropped_lesion_bbox
    ):
        return NotImplementedError

    def compute_discriminator_loss(self, negative, positive, desired_mask, lesion_bbox):
        return NotImplementedError

    def generate_fake(self, inputs, mask):
        coarse_image, fake_image = self.netG(inputs, mask)

        return coarse_image, fake_image

    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image.

    def discriminate(self, fake_image, real_image, mask):
        fake_concat = fake_image
        real_concat = real_image
        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)
        if self.opt.d_mask_in:
            mask_all = torch.cat([mask, mask], dim=0)
        else:
            mask_all = None

        discriminator_out = self.netD(fake_and_real, mask_all)

        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    def RCNN_loss(self, composed_image, ground_truth_bbox):
        targets = [
            {"boxes": torch.unsqueeze(bbox, 0), "labels": torch.LongTensor([1]).cuda()}
            for bbox in ground_truth_bbox
        ]  # label=1 for tumor class
        loss_dict = self.netDRCNN(composed_image, targets)

        _loss_sum = 0
        count = 0
        for key in loss_dict:
            count += 1
            _loss_sum += loss_dict[key]
        # TODO: think about what (sum of) loss is the correct one to return
        return _loss_sum

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[: tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2 :] for tensor in p])
        else:
            fake = pred[: pred.size(0) // 2]
            real = pred[pred.size(0) // 2 :]

        return fake, real

    def get_edges(self, t):
        edge = self.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0
