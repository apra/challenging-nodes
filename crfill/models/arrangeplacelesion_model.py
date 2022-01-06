import pdb
import torch
from models.placelesion_model import placelesionmodel
import util.util as util


class ArrangeplacelesionModel(placelesionmodel):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        placelesionmodel.modify_commandline_options(parser, is_train)
        parser.add_argument(
            "--load_base_g", type=str, required=False, help="load baseg"
        )
        parser.add_argument(
            "--load_base_d", type=str, required=False, help="load based"
        )
        parser.add_argument(
            "--lambda_ref",
            type=float,
            default=1,
            help="This is actually lambda in the paper, weight of the CR loss.",
        )
        parser.add_argument("--load_from_checkpoint", action="store_true")
        parser.add_argument(
            "--network_path",
            type=str,
            required=False,
            help="Path to the network file. of reconstruction",
        )
        parser.add_argument(
            "--latent_size", type=int, default=64, help="Latent size of VAE"
        )
        parser.add_argument(
            "--downsample",
            type=int,
            default=4,
            help="Downsamplings performed in encoder.",
        )
        parser.add_argument(
            "--sigma", type=float, default=0.06, help="Sigma of reconstruction"
        )
        parser.add_argument("--beta_kl", type=float, default=1, help="KL divergence")
        return parser

    def __init__(self, opt):
        super().__init__(opt)

    def save(self, epoch):
        #util.save_network(self.netG, "G", epoch, self.opt)
        util.save_network(self.place_lesion,"place_lesion", epoch, self.opt)
        util.save_network(self.netD, "D", epoch, self.opt)

    def create_optimizers(self, opt):
        print("Created optimizers")
        #G_params = [p for name, p in self.netG.named_parameters() if name.startswith("decoder")]
        G_params = [p for name, p in self.place_lesion.named_parameters()]
        # G_params = [p for name, p in self.netG.named_parameters() \
        #        if (not name.startswith("coarse"))]
        D_params = list(self.netD.parameters())
        DRCNN_params = self.netDRCNN.parameters()

        beta1, beta2 = opt.beta1, opt.beta2
        if opt.no_TTUR:
            G_lr, D_lr = opt.lr, opt.lr
        else:
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))
        optimizer_DRCNN = torch.optim.Adam(DRCNN_params, lr=0)

        return optimizer_G, optimizer_D, optimizer_DRCNN

    def forward(self, data, mode):
        negative, positive, mask, crop_bbox, lesion_bbox, cxr = self.preprocess_input(
            data
        )

        if mode == "generator":
            (
                g_loss,
                composed_image,
                _,
                _,
            ) = self.compute_generator_loss(
                negative, positive, mask, crop_bbox, lesion_bbox, cxr
            )
            generated = {
                "composed": composed_image,
            }
            return g_loss, negative, generated
        elif mode == "discriminator":
            d_loss = self.compute_discriminator_loss(negative, positive, mask)
            return d_loss, negative
        elif mode == "inference":
            with torch.no_grad():
                lesion = self.generate_fake(
                    negative, mask
                )
                composed_image = self.place_addition_on_cxr(lesion, negative, mask)
            return composed_image, negative
        else:
            raise ValueError("|mode| is invalid")

    def generate_fake(self, inputs, mask):
        lesion = self.netG.sample(samples=inputs.shape[0])

        return lesion

    def compute_discriminator_loss(self, negative, positive, mask):
        D_losses = {}
        if not self.opt.no_gan_loss:
            with torch.no_grad():
                lesion = self.generate_fake(
                    negative, mask
                )
                composed_image = self.place_addition_on_cxr(lesion, negative, mask)

            composed_image = composed_image.detach()
            composed_image.requires_grad_()

            pred_fake, pred_real = self.discriminate(composed_image, positive, mask)
            D_losses["D_Fake"] = self.criterionGAN(
                pred_fake, False, for_discriminator=True
            )
            D_losses["D_real"] = self.criterionGAN(
                pred_real, True, for_discriminator=True
            )

        return D_losses

    def place_using_bbox(self, base_image, crop_image, bbox):
        c_x1, c_y1, c_w, c_h = bbox.int()
        try:
            base_image[:, c_y1 : c_y1 + c_h, c_x1 : c_x1 + c_w] = (crop_image + 1) / 2
        except RuntimeError:
            print(crop_image)
            print(base_image)
            print(crop_image.shape)
            print(base_image.shape)

    def compute_generator_loss(
        self, negative, positive, mask, crop_bbox, lesion_bbox, cxr
    ):
        # if not self.opt.no_ganFeat_loss:
        #     raise NotImplementedError
        if self.opt.vgg_loss:
            raise NotImplementedError
        lesion = self.generate_fake(
            negative, mask
        )
        composed_image = self.place_addition_on_cxr(lesion, negative, mask)

        fake_cxr = cxr
        for i in range(len(crop_bbox)):
            self.place_using_bbox(fake_cxr[i], composed_image[i], crop_bbox[i])

        G_losses = self.g_image_loss(
            composed_image,
            negative,
            composed_image,
            positive,
            mask,
            fake_cxr,
            lesion_bbox,
        )

        return G_losses, composed_image, None, None
