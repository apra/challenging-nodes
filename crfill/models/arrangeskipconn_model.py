import pdb
import torch
from models.inpaintskipconn_model import InpaintskipconnModel
import util.util as util


class ArrangeskipconnModel(InpaintskipconnModel):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        InpaintskipconnModel.modify_commandline_options(parser, is_train)
        parser.add_argument('--load_base_g', type=str, required=False, help='load baseg')
        parser.add_argument('--load_base_d', type=str, required=False, help='load based')
        parser.add_argument('--lambda_ref', type=float, default=1,
                            help='This is actually lambda in the paper, weight of the CR loss.')
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        _, self.netD_aux, self.netDRCNN = self.initialize_networks(opt)
        if opt.continue_train:
            self.netD_aux = util.load_network(self.netD_aux, 'D_aux', opt.which_epoch, opt)
        if opt.load_base_g is not None:
            print(f"looad {opt.load_base_g}")
            self.netG.baseg = util.load_network_path(
                self.netG.baseg, opt.load_base_g)
        if opt.load_base_d is not None:
            print(f"looad {opt.load_base_d}")
            self.netD = util.load_network_path(
                self.netD, opt.load_base_d)

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)
        util.save_network(self.netD_aux, 'D_aux', epoch, self.opt)

    def create_optimizers(self, opt):
        G_params = self.netG.get_param_list(opt.update_part)
        # G_params = [p for name, p in self.netG.named_parameters() \
        #        if (not name.startswith("coarse"))]
        if opt.isTrain:
            D_params = list(self.netD.parameters()) + \
                       list(self.netD_aux.parameters())

        beta1, beta2 = opt.beta1, opt.beta2
        if opt.no_TTUR:
            G_lr, D_lr = opt.lr, opt.lr
        else:
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        return optimizer_G, optimizer_D

    def forward(self, data, mode):
        negative, positive, mask, crop_bbox, lesion_bbox, cxr = self.preprocess_input(data)

        if mode == 'generator':
            g_loss, composed_image, composed_image_aux, recon_aux = self.compute_generator_loss(negative, positive,
                                                                                                mask, crop_bbox, lesion_bbox, cxr)
            generated = {
                'composed': composed_image,
                'composed_aux': composed_image_aux,
                **recon_aux
            }
            return g_loss, negative, generated
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(negative, positive, mask)
            return d_loss, negative
        elif mode == 'inference':
            with torch.no_grad():
                coarse_image, fake_image, aux_image, recon_aux = self.generate_fake(
                    negative, mask)
                composed_image = self.place_addition_on_cxr(fake_image, negative, mask)
            return composed_image, negative
        else:
            raise ValueError("|mode| is invalid")

    def generate_fake(self, inputs, mask):
        coarse_image, fake_image, aux_image, recon_aux = self.netG(inputs, mask)

        return coarse_image, fake_image, aux_image, recon_aux

    def compute_discriminator_loss(self, negative, positive, mask):
        D_losses = {}
        if not self.opt.no_gan_loss:
            with torch.no_grad():
                coarse_image, fake_image, aux_image, recon_aux = self.generate_fake(negative, mask)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()
            aux_image = aux_image.detach()
            aux_image.requires_grad_()
            composed_image = self.place_addition_on_cxr(fake_image, negative, mask)
            composed_aux = self.place_addition_on_cxr(aux_image, negative, mask)

            pred_fake, pred_real = self.discriminate(composed_image, positive, mask)
            D_losses['D_Fake'] = self.criterionGAN(pred_fake, False,
                                                   for_discriminator=True)
            D_losses['D_real'] = self.criterionGAN(pred_real, True,

                                                   for_discriminator=True)
            _netD = self.netD
            self.netD = self.netD_aux
            pred_fake, pred_real = self.discriminate(composed_aux, positive, mask)
            D_losses['D_Fake_aux'] = self.criterionGAN(pred_fake, False,
                                                       for_discriminator=True)
            D_losses['D_real_aux'] = self.criterionGAN(pred_real, True,
                                                       for_discriminator=True)
            self.netD = _netD

        return D_losses

    def compute_generator_loss(self, negative, positive, mask, crop_bbox, lesion_bbox, cxr):
        # if not self.opt.no_ganFeat_loss:
        #     raise NotImplementedError
        if self.opt.vgg_loss:
            raise NotImplementedError
        coarse_addition, additive_generation, aux_image, recon_aux = self.generate_fake(negative, mask)
        coarse_image = self.place_addition_on_cxr(coarse_addition, negative, mask)
        composed_image = self.place_addition_on_cxr(additive_generation, negative, mask)

        fake_cxr = cxr
        for i in range(len(crop_bbox)):
            c_x1, c_y1, c_w, c_h = crop_bbox[i].int()
            try:
                fake_cxr[i, :, c_y1:c_y1 + c_h, c_x1:c_x1 + c_w] = composed_image[i]
            except RuntimeError:
                print(composed_image[i])
                print(fake_cxr[i])
                print(composed_image[i].shape)
                print(fake_cxr[i].shape)

        G_losses = self.g_image_loss(coarse_image, negative, composed_image, positive, mask, fake_cxr, lesion_bbox)

        composed_image_aux = self.place_addition_on_cxr(aux_image, negative, mask)
        _netD = self.netD
        self.netD = self.netD_aux
        G_losses_aux = self.g_image_loss(None, negative, composed_image_aux, positive, mask, fake_cxr, lesion_bbox)
        self.netD = _netD
        for k, v in G_losses_aux.items():
            G_losses[k + "_aux"] = v * self.opt.lambda_ref
        return G_losses, composed_image, composed_image_aux, recon_aux
