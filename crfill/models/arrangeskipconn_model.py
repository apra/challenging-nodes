import pdb
import torch
from models.inpaintskipconn_model import InpaintskipconnModel
import util.util as util
from PIL import Image


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
        _, self.netD_aux, self.netDRCNN_aux = self.initialize_networks(opt)
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
        negative, positive, mask, crop_bbox, lesion_bbox, cxr, negative_masked = self.preprocess_input(data)

        if mode == 'generator':
            g_loss, composed_image, composed_image_aux, recon_aux = self.compute_generator_loss(negative, positive,
                                                                                                mask, crop_bbox,
                                                                                                lesion_bbox, cxr, negative_masked)
            generated = {
                'composed': composed_image,
                'composed_aux': composed_image_aux,
                **recon_aux
            }
            return g_loss, negative, generated
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(negative, positive, mask, negative_masked)
            return d_loss, negative
        elif mode == 'inference':
            with torch.no_grad():
                coarse_image, fake_image, aux_image, recon_aux = self.generate_fake(
                    negative_masked, mask)
                composed_image = self.place_addition_on_cxr(fake_image, negative, mask)
            return composed_image, negative_masked
        else:
            raise ValueError("|mode| is invalid")

    def generate_fake(self, inputs, mask):
        coarse_image, fake_image, aux_image, recon_aux = self.netG(inputs, mask)

        return coarse_image, fake_image, aux_image, recon_aux

    def compute_discriminator_loss(self, negative, positive, mask, negative_masked):
        D_losses = {}
        if not self.opt.no_gan_loss:
            with torch.no_grad():
                coarse_image, fake_image, aux_image, recon_aux = self.generate_fake(negative_masked, mask)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()
            aux_image = aux_image.detach()
            aux_image.requires_grad_()
            composed_image = self.place_addition_on_cxr(fake_image, negative, mask)
            composed_aux = self.place_addition_on_cxr(aux_image, negative, mask)


            if self.opt.mask_pos_discriminator:
                composed_masked = composed_image * mask
                positive_masked = positive * mask
                pred_fake, pred_real = self.discriminate(composed_masked, positive_masked, mask)
                D_losses['D_Fake'] = self.criterionGAN(pred_fake, False,
                                                       for_discriminator=True)
                D_losses['D_real'] = self.criterionGAN(pred_real, True,
                                                       for_discriminator=True)
            else:
                if self.opt.mask_pos_discriminator:
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

    def place_using_bbox(self, base_image, crop_image, bbox):
        c_x1, c_y1, c_w, c_h = bbox.int()
        try:
            base_image[:, c_y1:c_y1 + c_h, c_x1:c_x1 + c_w] = (crop_image + 1) / 2
        except RuntimeError:
            print(crop_image)
            print(base_image)
            print(crop_image.shape)
            print(base_image.shape)

    def compute_generator_loss(self, negative, positive, mask, crop_bbox, lesion_bbox, cxr, negative_masked):
        # if not self.opt.no_ganFeat_loss:
        #     raise NotImplementedError
        if self.opt.vgg_loss:
            raise NotImplementedError
        coarse_addition, additive_generation, aux_image, recon_aux = self.generate_fake(negative_masked, mask)
        coarse_image = self.place_addition_on_cxr(coarse_addition, negative, mask)
        composed_image = self.place_addition_on_cxr(additive_generation, negative, mask)

        fake_cxr = cxr
        for i in range(len(crop_bbox)):
            self.place_using_bbox(fake_cxr[i], composed_image[i], crop_bbox[i])

        G_losses = self.g_image_loss(coarse_image, negative, composed_image, positive, mask, fake_cxr, lesion_bbox)

        composed_image_aux = self.place_addition_on_cxr(aux_image, negative, mask)
        _netD = self.netD
        self.netD = self.netD_aux
        G_losses_aux = self.g_image_loss(None, negative, composed_image_aux, positive, mask, None, lesion_bbox)
        self.netD = _netD
        for k, v in G_losses_aux.items():
            G_losses[k + "_aux"] = v * self.opt.lambda_ref
        return G_losses, composed_image, composed_image_aux, recon_aux
