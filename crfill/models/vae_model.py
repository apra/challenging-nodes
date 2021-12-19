import math
from dataclasses import dataclass
from typing import List, Dict, Optional

import torch
import torch.distributions as dists
from torch import nn
from torch.nn import functional as F
import util.util as util


class Interpolate(nn.Module):
    """Wrapper for torch.nn.functional.interpolate."""

    def __init__(self, size=None, scale=None, mode="bilinear", align_corners=False):
        super().__init__()
        assert (size is None) == (scale is not None)
        self.size = size
        self.scale = scale
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        out = F.interpolate(
            x,
            size=self.size,
            scale_factor=self.scale,
            mode=self.mode,
            align_corners=self.align_corners,
        )
        return out


class ResidualBlock(nn.Module):
    def __init__(
        self,
        n_channels,
        *,
        num_layers=2,
        kernel_size=3,
        dilation=1,
        groups=1,
        rezero=True
    ):
        super(ResidualBlock, self).__init__()
        ch = n_channels
        assert kernel_size % 2 == 1
        pad = kernel_size // 2
        layers = []
        for i in range(num_layers):
            layers.extend(
                [
                    nn.LeakyReLU(1e-2),
                    nn.Conv2d(
                        ch,
                        ch,
                        kernel_size=kernel_size,
                        padding=pad,
                        dilation=dilation,
                        groups=groups,
                    ),
                ]
            )
            # if i < num_layers - 1:
            #     layers.append(nn.Dropout2d(0.2))
        self.net = nn.Sequential(*layers)
        if rezero:
            self.gate = nn.Parameter(
                torch.zeros(1, dtype=torch.float32), requires_grad=True
            )
        else:
            self.gate = 1.0

    def forward(self, inputs):
        return inputs + self.net(inputs) * self.gate


# def safe_channel_change(channel_size_per_layer, layer, encoder):
#     """
#     Print some safety messages if the number of channels is increasing in a decoder or
#     decreasing in an encoder, which is un-orthodox.
#     """
#     if encoder and channel_size_per_layer[layer] < channel_size_per_layer[layer - 1]:
#         print("Decreasing the number of channels in the encoder!")
#     elif (
#         not encoder
#         and channel_size_per_layer[layer] > channel_size_per_layer[layer - 1]
#     ):
#         print("Increasing the number of channels in the decoder!")


def log_residual_stack_structure(
    channel_size_per_layer: List[int],
    layers_per_block_per_layer: List[int],
    downsample: int,
    num_layers_per_resolution: List[int],
    encoder: bool = True,
):
    print("Creating structure with {} downsamples.".format(downsample))
    layers = []

    assert len(channel_size_per_layer) == sum(num_layers_per_resolution)
    assert downsample <= len(num_layers_per_resolution)

    layer = 0

    for block_num, num_layers in enumerate(num_layers_per_resolution):
        for _ in range(num_layers):
            layers.append(
                "Residual Block with "
                "{} channels and "
                "{} layers.".format(
                    channel_size_per_layer[layer], layers_per_block_per_layer[layer]
                )
            )
            layer += 1
            # if it's not the last layer, check if the next one has more channels and connect them
            # using a conv layer
            if layer < len(channel_size_per_layer):
                if channel_size_per_layer[layer] != channel_size_per_layer[layer - 1]:
                    layers.append(
                        "Con2d layer with "
                        "{} input channels and "
                        "{} output channels".format(
                            channel_size_per_layer[layer - 1],
                            channel_size_per_layer[layer],
                        )
                    )
                    # safe_channel_change(channel_size_per_layer, layer, encoder)

        # after the residual block, check if down-sampling (or up-sampling) is required
        if encoder:
            if downsample > 0:
                layers.append("Avg Pooling layer.")
                downsample -= 1
        else:
            if block_num + downsample > (len(num_layers_per_resolution) - 1):
                layers.append("Interpolation layer.")

    return layers


def build_residual_stack(
    channel_size_per_layer: List[int],
    layers_per_block_per_layer: List[int],
    downsample: int,
    num_layers_per_resolution: List[int],
    encoder: bool = True,
):
    print(
        "\n".join(
            log_residual_stack_structure(
                channel_size_per_layer=channel_size_per_layer,
                layers_per_block_per_layer=layers_per_block_per_layer,
                downsample=downsample,
                num_layers_per_resolution=num_layers_per_resolution,
                encoder=encoder,
            )
        )
    )
    layers = []

    assert len(channel_size_per_layer) == sum(num_layers_per_resolution)
    assert downsample <= len(num_layers_per_resolution)

    layer = 0

    for block_num, num_layers in enumerate(num_layers_per_resolution):
        for _ in range(num_layers):
            # add a residual block with the required number of channels and layers
            layers.append(
                ResidualBlock(
                    channel_size_per_layer[layer],
                    num_layers=layers_per_block_per_layer[layer],
                )
            )
            layer += 1
            # if it's not the last layer, check if the next one has more channels and connect them
            # using a conv layer
            if layer < len(channel_size_per_layer):
                if channel_size_per_layer[layer] != channel_size_per_layer[layer - 1]:
                    # safe_channel_change(channel_size_per_layer, layer, encoder)

                    in_channels = channel_size_per_layer[layer - 1]
                    out_channels = channel_size_per_layer[layer]
                    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))

        # after the residual blocks, check if down-sampling (or up-sampling) is required
        if encoder:
            if downsample > 0:
                layers.append(
                    nn.AvgPool2d(kernel_size=2, stride=2),
                )
                downsample -= 1
        else:
            if block_num + downsample > (len(num_layers_per_resolution) - 1):
                layers.append(Interpolate(scale=2))

    return layers


class Encoder(torch.nn.Module):
    def __init__(
        self,
        channel_size_per_layer: List[int],
        layers_per_block_per_layer: List[int],
        latent_size: int,
        width: int,
        height: int,
        num_layers_per_resolution,
        mlp_hidden_size: int = 512,
        channel_size: int = 64,
        input_channels: int = 3,
        downsample: int = 4,
    ):
        super().__init__()
        self.latent_size = latent_size

        # compute final width and height of feature maps
        inner_width = width // (2 ** downsample)
        inner_height = height // (2 ** downsample)

        # conv layers
        layers = [
            nn.Conv2d(input_channels, channel_size, 5, padding=2, stride=2),
            nn.LeakyReLU(),
        ]
        layers.extend(
            build_residual_stack(
                channel_size_per_layer=channel_size_per_layer,
                layers_per_block_per_layer=layers_per_block_per_layer,
                downsample=downsample - 1,
                num_layers_per_resolution=num_layers_per_resolution,
                encoder=True,
            )
        )

        mlp_input_size = channel_size_per_layer[-1] * inner_width * inner_height

        # fully connected MLP with two hidden layers
        layers.extend(
            [
                nn.Flatten(),
                nn.LeakyReLU(),
                nn.Linear(mlp_input_size, mlp_hidden_size),
                nn.LeakyReLU(),
                nn.LayerNorm(mlp_hidden_size),
            ]
        )

        self.net = nn.Sequential(*layers)
        self.mean = nn.Linear(mlp_hidden_size, latent_size)
        self.logvar = nn.Linear(mlp_hidden_size, latent_size)
        torch.nn.init.normal_(self.logvar.weight, std=0.01)
        torch.nn.init.zeros_(self.logvar.bias)
        # torch.nn.init.normal_(self.mean.weight, std=0.02)
        # torch.nn.init.zeros_(self.mean.bias)

    def forward(self, x):
        q_activations = self.net(x)
        mean = self.mean(q_activations)
        logvar = self.logvar(q_activations)
        sigma = (logvar * 0.5).exp()
        prior_dist = dists.Normal(0.0, 1.0)

        latent_normal = dists.Normal(mean, sigma)
        kl = dists.kl_divergence(latent_normal, prior_dist)  # [Batch size, Latent size]
        assert kl.shape == (x.shape[0], self.latent_size)
        kl = kl.sum(dim=1)  # [Batch size]
        z = latent_normal.rsample()  # [Batch size, Latent size]
        return {"z": z, "kl": kl, "q_mean": mean, "q_logvar": logvar, "q_sigma": sigma}


class MLPDecoder(torch.nn.Module):
    def __init__(
        self,
        latent_size: int,
        width: int,
        height: int,
        channel_size_per_layer: List[int] = (256, 256, 256, 256, 128, 128, 64, 64),
        layers_per_block_per_layer: List[int] = (2, 2, 2, 2, 2, 2, 2, 2),
        num_layers_per_resolution: List[int] = (2, 2, 2, 2),
        input_channels: int = 3,
        downsample: Optional[int] = 4,
        mlp_hidden_size: Optional[int] = 512,
    ):
        super().__init__()

        # compute final width and height of feature maps
        inner_width = width // (2 ** downsample)
        inner_height = height // (2 ** downsample)

        mlp_input_size = channel_size_per_layer[0] * inner_width * inner_height

        # fully connected MLP with two hidden layers
        layers = []
        layers.extend(
            [
                nn.Linear(latent_size, mlp_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(mlp_hidden_size, mlp_input_size),
                nn.Unflatten(
                    1,
                    unflattened_size=(
                        channel_size_per_layer[0],
                        inner_height,
                        inner_width,
                    ),
                ),
                # B, 64*4, 4, 4
            ]
        )

        # conv layers
        layers.extend(
            build_residual_stack(
                channel_size_per_layer=channel_size_per_layer,
                layers_per_block_per_layer=layers_per_block_per_layer,
                downsample=downsample,
                num_layers_per_resolution=num_layers_per_resolution,
                encoder=False,
            )
        )
        layers.append(nn.LeakyReLU())

        final_conv = nn.Conv2d(channel_size_per_layer[-1], input_channels, 5, padding=2)
        torch.nn.init.zeros_(final_conv.bias)
        torch.nn.init.trunc_normal_(final_conv.weight, std=0.01)
        layers.append(final_conv)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class BroadcastDecoder(torch.nn.Module):
    def __init__(
        self,
        latent_size: int,
        width: int,
        height: int,
        broadcast_size: Optional[int] = 8,
        channel_size_per_layer: List[int] = (256, 256, 256, 256, 128, 128, 64, 64),
        layers_per_block_per_layer: List[int] = (2, 2, 2, 2, 2, 2, 2, 2),
        num_layers_per_resolution: List[int] = (2, 2, 2, 2),
        input_channels: int = 3,
    ):
        super().__init__()
        downsample = math.ceil(math.log2(width / broadcast_size))

        print(
            "Broadcast: {} \nWidth:{} \nHeight:{} \nDownsample: {}".format(
                broadcast_size, width, height, downsample
            )
        )

        # compute final width and height of feature maps
        inner_width = width // (2 ** downsample)
        inner_height = height // (2 ** downsample)

        self.h_broadcast = inner_height
        self.w_broadcast = inner_width

        ys = torch.linspace(-1, 1, self.h_broadcast)
        xs = torch.linspace(-1, 1, self.w_broadcast)
        ys, xs = torch.meshgrid(ys, xs, indexing="ij")
        coord_map = torch.stack((ys, xs)).unsqueeze(0)
        self.register_buffer("coord_map_const", coord_map)

        layers = []
        # conv layers
        layers.append(
            nn.Conv2d(
                latent_size + 2, channel_size_per_layer[0], 5, padding=2, stride=1
            )
        )
        layers.extend(
            build_residual_stack(
                channel_size_per_layer=channel_size_per_layer,
                layers_per_block_per_layer=layers_per_block_per_layer,
                downsample=downsample,
                num_layers_per_resolution=num_layers_per_resolution,
                encoder=False,
            )
        )
        layers.append(nn.LeakyReLU())

        final_conv = nn.Conv2d(channel_size_per_layer[-1], input_channels, 5, padding=2)
        torch.nn.init.zeros_(final_conv.bias)
        torch.nn.init.trunc_normal_(final_conv.weight, std=0.01)
        layers.append(final_conv)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        batch_size = x.shape[0]
        z_tiled = (
            x.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(batch_size, x.shape[1], self.h_broadcast, self.w_broadcast)
        )
        coord_map = self.coord_map_const.expand(
            batch_size, 2, self.h_broadcast, self.w_broadcast
        )
        inp = torch.cat((z_tiled, coord_map), 1)
        return self.net(inp)


class DecoderParams(Dict):
    architecture_type: str
    mlp_hidden_size: int
    latent_size: int
    broadcast_size: int
    channel_size_per_layer: List[int]
    layers_per_block_per_layer: List[int]
    num_layers_per_resolution: List[int]


class EncoderParams(Dict):
    latent_size: int
    broadcast_size: int
    channel_size_per_layer: List[int]
    layers_per_block_per_layer: List[int]
    num_layers_per_resolution: List[int]
    downsample: int
    mlp_hidden_size: int


@dataclass(eq=False, repr=False)  # for compatibility reasons
class vaemodel(nn.Module):

    width: int
    height: int
    latent_size: int
    num_slots: int
    beta_kl: float
    opt: Dict

    decoder_params: DecoderParams
    encoder_params: EncoderParams
    input_channels: int = 1

    sigma: float = 0.09

    name: str = "baseline-vae"

    def __post_init__(self):
        super().__init__()
        self.full_latent_size = self.latent_size * self.num_slots
        self.encoder_params.update(
            width=self.width,
            height=self.height,
            input_channels=self.input_channels,
            latent_size=self.full_latent_size,
        )
        self.encoder = Encoder(**self.encoder_params)
        self.decoder_params.update(
            width=self.width,
            height=self.height,
            input_channels=self.input_channels,
            latent_size=self.full_latent_size,
        )

        if self.decoder_params["architecture_type"] == "mlp":
            del self.decoder_params["architecture_type"]
            del self.decoder_params["broadcast_size"]
            self.decoder_params["downsample"] = self.encoder_params["downsample"]
            self.decoder = MLPDecoder(**self.decoder_params)
        elif self.decoder_params["architecture_type"] == "broadcast":
            del self.decoder_params["architecture_type"]
            del self.decoder_params["mlp_hidden_size"]
            self.decoder = BroadcastDecoder(**self.decoder_params)
        else:
            raise ValueError(
                "Only 'mlp' or 'broadcast' architecture_type "
                "are allowed, but it was: {}.".format(
                    self.decoder_params["architecture_type"]
                )
            )
        self.register_buffer("fake_mask", torch.ones((1, 1, 1, 1, 1)))

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument(
            "--latent_size", type=int, default=64, help="Latent size of VAE"
        )
        parser.add_argument(
            "--sigma", type=float, default=0.06, help="Sigma of reconstruction"
        )
        parser.add_argument(
            "--beta_kl", type=float, default=1, help="KL divergence"
        )
        return parser

    def save(self, epoch):
        util.save_network(self, "G", epoch, self.opt)

    def forward(self, x):
        forward_out = self.forward_vae_slots(x)

        loss_out = self._calc_loss(forward_out["kl_z"], forward_out["log_p_x"])

        mask = self.fake_mask.expand(
            forward_out["x_recon"].shape[0],
            1,
            1,
            forward_out["x_recon"].shape[2],
            forward_out["x_recon"].shape[3],
        )

        gate_values = {n: p for n, p in self.named_parameters() if n.endswith(".gate")}
        return {
            "loss": loss_out["loss"] / (x.shape[1] * x.shape[2] * x.shape[3]),
            "neg_log_p_x": loss_out["neg_log_p_xs"],
            "kl_latent": forward_out["kl_z"].mean(),
            "latent_means": forward_out["latent_means"],
            "latent_sigmas": forward_out["latent_sigmas"],
            "latent_logvars": forward_out["latent_logvars"],
            "z": forward_out["z"],
            "reconstructed": forward_out["x_recon"].clamp(0.0, 1.0),
            "slot": forward_out["x_recon"].unsqueeze(1),
            "mask": mask,
            "mse": ((x - forward_out["x_recon"]) ** 2).mean(),
            "representation": forward_out["latent_means"],
            **gate_values,
        }

    def forward_vae_slots(self, x):
        encoder_out = self.encoder(x)
        log_p_x, x_recon = self._decode(x, encoder_out["z"], self.sigma)

        return {
            "kl_z": encoder_out["kl"],
            "x_recon": x_recon,
            "log_p_x": log_p_x,
            "z": encoder_out["z"],
            "latent_means": encoder_out["q_mean"],
            "latent_sigmas": encoder_out["q_sigma"],
            "latent_logvars": encoder_out["q_logvar"],
        }

    def _calc_loss(self, kl_z, log_p_xs):
        neg_log_p_xs = -log_p_xs.mean(dim=0).sum()
        neg_elbo = neg_log_p_xs + self.beta_kl * kl_z.mean()

        return {"loss": neg_elbo, "neg_log_p_xs": neg_log_p_xs}

    def _decode(self, x, z, sigma):
        # x [Batch size, channels, height, width]
        decoder_output = self.decoder(z)
        # x_recon = decoder_output
        # x_recon = torch.clamp(decoder_output, 0, 1)
        x_recon = decoder_output.sigmoid()
        dist = dists.Normal(x_recon, sigma)
        log_p_x = dist.log_prob(x)
        return log_p_x, x_recon
