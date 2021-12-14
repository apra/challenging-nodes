import os.path
import pdb
import torch
from models.inpaintdoubledisc_model import InpaintdoublediscModel
import util.util as util
from pathlib import Path


class ArrangedoublediscModel(InpaintdoublediscModel):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        InpaintdoublediscModel.modify_commandline_options(parser, is_train)
        parser.add_argument('--load_base_g', type=str, default=None, help='load baseg')
        parser.add_argument('--load_base_d', type=str, default=None, help='load based')
        parser.add_argument('--lambda_ref', type=float, default=1, help='This is actually lambda in the paper, weight of the CR loss.')
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        _, self.netD_aux, _ = self.initialize_networks(opt)

        if opt.continue_train:
            self.netD_aux = util.load_network(self.netD_aux, 'D_aux', opt.which_epoch, opt)
        if opt.load_base_g is not None:
            print(f"looad {opt.load_base_g}")
            self.netG = util.load_network_path(
                    self.netG, opt.load_base_g)
        if opt.load_base_d is not None:
            print(f"looad {opt.load_base_d}")
            self.netD = util.load_network_path(
                    self.netD, opt.load_base_d)
            print("loading aux")
            aux_path = str(opt.load_base_d)[:-4] + '_aux.pth'
            self.netD_aux = util.load_network_path(
                self.netD_aux, aux_path)

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)
        util.save_network(self.netD_aux, 'D_aux', epoch, self.opt)

    def create_optimizers(self, opt):
        G_params = self.netG.get_param_list(opt.update_part)
        #G_params = [p for name, p in self.netG.named_parameters() \
        #        if (not name.startswith("coarse"))]
        if opt.isTrain:
            D_params = list(self.netD.parameters()) + \
                    list(self.netD_aux.parameters())
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
        inputs, real_image, mask, full_image, full_image_crop_bbox, full_image_bbox = self.preprocess_input(data)

        if mode == 'generator':
            g_loss, composed_image, composed_image_aux, recon_aux =\
                    self.compute_generator_loss(inputs, real_image, mask, full_image, full_image_crop_bbox, full_image_bbox)
            generated = {
                    'composed': composed_image,
                    'composed_aux': composed_image_aux,
                    **recon_aux
                    }
            return g_loss, inputs, generated
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(
                inputs, real_image, mask)
            return d_loss, inputs
        elif mode == 'inference':
            with torch.no_grad():
                coarse_image, fake_image, aux_image, recon_aux = self.generate_fake(inputs, mask)
                composed_image = fake_image*mask + inputs*(1-mask)
            return composed_image, inputs
        else:
            raise ValueError("|mode| is invalid")


    def generate_fake(self, inputs, mask):
        coarse_image, fake_image, aux_image, recon_aux = self.netG(inputs, mask)

        return coarse_image, fake_image, aux_image, recon_aux

    def compute_discriminator_loss(self, inputs, real_image, mask):
        D_losses = {}
        if not self.opt.no_gan_loss:
            with torch.no_grad():
                coarse_image, fake_image, aux_image, recon_aux = self.generate_fake(
                        inputs, mask)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()
            aux_image = aux_image.detach()
            aux_image.requires_grad_()
            composed_image = fake_image*mask + inputs*(1-mask)
            composed_aux = aux_image*mask + inputs*(1-mask)

            pred_fake, pred_real = self.discriminate(
                composed_image, real_image, mask)
            D_losses['D_Fake'] = self.criterionGAN(pred_fake, False,
                                                   for_discriminator=True)
            D_losses['D_real'] = self.criterionGAN(pred_real, True,

                                                   for_discriminator=True)
            _netD = self.netD
            self.netD = self.netD_aux
            pred_fake, pred_real = self.discriminate(
                composed_aux, real_image, mask)
            D_losses['D_Fake_aux'] = self.criterionGAN(pred_fake, False,
                                                   for_discriminator=True)
            D_losses['D_real_aux'] = self.criterionGAN(pred_real, True,
                                                   for_discriminator=True)
            self.netD = _netD

        return D_losses

    def compute_generator_loss(self, inputs, real_image, mask, full_image, full_image_crop_bbox, full_image_bbox):
        # if not self.opt.no_ganFeat_loss:
        #     raise NotImplementedError
        if self.opt.vgg_loss:
            raise NotImplementedError
        coarse_image, fake_image, aux_image, recon_aux = self.generate_fake(
                inputs, mask)
        composed_image = fake_image*mask + inputs*(1-mask)

        composed_full_image = full_image
        for i in range(len(full_image_crop_bbox)):
            c_x1, c_y1, c_w, c_h = full_image_crop_bbox[i].int()
            try:
                composed_full_image[i, :, c_y1:c_y1+c_h, c_x1:c_x1+c_w] = (composed_image[i] + 1) / 2
            except RuntimeError:
                print(composed_image[i])
                print(composed_full_image[i])
                print(composed_image[i].shape)
                print(composed_full_image[i].shape)


        G_losses = self.g_image_loss(coarse_image, fake_image, composed_image, real_image, mask, composed_full_image, full_image_bbox)

        composed_image_aux = aux_image*mask + inputs*(1-mask)
        _netD = self.netD
        self.netD = self.netD_aux
        G_losses_aux = self.g_image_loss(None, aux_image, composed_image_aux, real_image, mask, None, None)
        self.netD = _netD
        for k,v in G_losses_aux.items():
            G_losses[k+"_aux"] = v*self.opt.lambda_ref
        return G_losses, composed_image, composed_image_aux, recon_aux
