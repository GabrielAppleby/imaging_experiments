import itertools
from typing import Tuple

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import nn as nn
from torch.nn import Identity
#from torchinfo import summary


class BatchNormSwish(nn.Module):
    def __init__(self, num_channels):
        super(BatchNormSwish, self).__init__()
        self._seq = nn.Sequential(nn.BatchNorm2d(num_channels), nn.SiLU())

    def forward(self, x):
        return self._seq(x)


class WeightNormedConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=True,
                 groups=1):
        super(WeightNormedConv2d, self).__init__()
        self._conv = nn.utils.spectral_norm(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=(kernel_size, kernel_size),
                      stride=(stride, stride),
                      padding=(padding, padding),
                      bias=bias,
                      groups=groups))

    def forward(self, x):
        return self._conv(x)


class FactorizedReduce(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FactorizedReduce, self).__init__()
        assert out_channels % 2 == 0
        reduced_out_channels = out_channels // 4
        self.silu = nn.SiLU()
        self.conv_1 = WeightNormedConv2d(in_channels,
                                         reduced_out_channels,
                                         1,
                                         stride=2,
                                         padding=0,
                                         bias=True)
        self.conv_2 = WeightNormedConv2d(in_channels,
                                         reduced_out_channels,
                                         1,
                                         stride=2,
                                         padding=0,
                                         bias=True)
        self.conv_3 = WeightNormedConv2d(in_channels,
                                         reduced_out_channels,
                                         1,
                                         stride=2,
                                         padding=0,
                                         bias=True)
        self.conv_4 = WeightNormedConv2d(in_channels,
                                         out_channels - 3 * reduced_out_channels,
                                         1,
                                         stride=2,
                                         padding=0,
                                         bias=True)

    def forward(self, x):
        x = self.silu(x)
        conv1 = self.conv_1(x)
        conv2 = self.conv_2(x[:, :, 1:, 1:])
        conv3 = self.conv_3(x[:, :, :, 1:])
        conv4 = self.conv_4(x[:, :, 1:, :])
        x = torch.cat([conv1, conv2, conv3, conv4], dim=1)
        return x


class BatchNormSwishWeightNormedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(BatchNormSwishWeightNormedConv2d, self).__init__()
        self._seq = nn.Sequential(BatchNormSwish(in_channels),
                                  WeightNormedConv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=stride,
                                                     padding=1,
                                                     bias=True))

    def forward(self, x):
        return self._seq(x)


class WeightNormedConv2dBatchNormSwish(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size=1, groups=1):
        super(WeightNormedConv2dBatchNormSwish, self).__init__()
        self._seq = nn.Sequential(WeightNormedConv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=kernel_size,
                                                     stride=stride,
                                                     padding=1,
                                                     bias=True,
                                                     groups=groups),
                                  BatchNormSwish(out_channels))

    def forward(self, x):
        return self._seq(x)


class SELayer(nn.Module):
    """https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py"""

    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class EncoderConvSEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(EncoderConvSEBlock, self).__init__()
        self._seq = nn.Sequential(BatchNormSwishWeightNormedConv2d(in_channels,
                                                                   out_channels,
                                                                   stride),
                                  BatchNormSwishWeightNormedConv2d(out_channels, out_channels, 1),
                                  SELayer(out_channels))

    def forward(self, x):
        return self._seq(x)


class DecoderConvSEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, ex=6, upsample=False):
        super(DecoderConvSEBlock, self).__init__()
        hidden_channels = in_channels * ex
        upsample_module = nn.ModuleList()
        if upsample:
            upsample_module.append(nn.UpsamplingNearest2d(scale_factor=2))
        self._seq = nn.Sequential(*upsample_module,
                                  nn.BatchNorm2d(in_channels),
                                  WeightNormedConv2dBatchNormSwish(in_channels, hidden_channels, 1),
                                  WeightNormedConv2dBatchNormSwish(hidden_channels,
                                                                   hidden_channels,
                                                                   1,
                                                                   5,
                                                                   hidden_channels),
                                  nn.Conv2d(hidden_channels,
                                            out_channels,
                                            kernel_size=(1, 1),
                                            stride=(stride, stride),
                                            padding=(0, 0),
                                            bias=False),
                                  nn.BatchNorm2d(out_channels),
                                  SELayer(out_channels))

    def forward(self, x):
        return self._seq(x)


class Upsample(nn.Module):
    def __init__(self,  scale_factor=2):
        super(Upsample, self).__init__()
        self._scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self._scale_factor, mode='bilinear', align_corners=True)


class DecoderUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, stride, ex=6):
        super(DecoderUpsample, self).__init__()
        self._conv = DecoderResBlock(in_channels,
                                     out_channels,
                                     stride,
                                     nn.Sequential(Upsample(),
                                                   WeightNormedConv2d(in_channels,
                                                                      out_channels,
                                                                      kernel_size=1,
                                                                      padding=0)),
                                     ex=ex,
                                     upsample=True)

    def forward(self, x):
        return self._conv(x)


class EncoderResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, skip_connection):
        super(EncoderResBlock, self).__init__()
        self._convseblock = EncoderConvSEBlock(in_channels, out_channels, stride)
        self._skip_connection = skip_connection

    def forward(self, x):
        return self._convseblock(x) + self._skip_connection(x)


class DecoderResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, skip_connection, ex=6, upsample=False):
        super(DecoderResBlock, self).__init__()
        self._convseblock = DecoderConvSEBlock(in_channels, out_channels, stride, ex=ex, upsample=upsample)
        self._skip_connection = skip_connection

    def forward(self, x):
        return self._convseblock(x) + self._skip_connection(x)


class EncoderIdentityResBlock(nn.Module):
    def __init__(self, num_channels):
        super(EncoderIdentityResBlock, self).__init__()
        self._resblock = EncoderResBlock(num_channels, num_channels, 1, Identity())

    def forward(self, x):
        return self._resblock(x)


class DecoderIdentityResBlock(nn.Module):
    def __init__(self, num_channels, ex=6):
        super(DecoderIdentityResBlock, self).__init__()
        self._resblock = DecoderResBlock(num_channels, num_channels, 1, Identity(), ex=ex)

    def forward(self, x):
        return self._resblock(x)


class FactorizedReduceResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FactorizedReduceResBlock, self).__init__()
        self._resblock = EncoderResBlock(in_channels, out_channels, 2,
                                         FactorizedReduce(in_channels, out_channels))

    def forward(self, x):
        return self._resblock(x)


class Stem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Stem, self).__init__()
        self._stem = WeightNormedConv2d(in_channels,
                                        out_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        bias=True)

    def forward(self, x):
        return self._stem(x)


class Preprocess(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Preprocess, self).__init__()
        self._seq = nn.Sequential(EncoderIdentityResBlock(in_channels),
                                  FactorizedReduceResBlock(in_channels, out_channels))

    def forward(self, x):
        return self._seq(x)


class Postprocess(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Postprocess, self).__init__()
        self._seq = nn.Sequential(DecoderUpsample(in_channels, out_channels, stride=1, ex=3),
                                  DecoderIdentityResBlock(out_channels, ex=3))

    def forward(self, x):
        return self._seq(x)

class EncoderGroup(nn.Module):
    def __init__(self, channels):
        super(EncoderGroup, self).__init__()
        self._seq = nn.Sequential(EncoderIdentityResBlock(channels),
                                  EncoderIdentityResBlock(channels))

    def forward(self, x):
        return self._seq(x)


class DecoderGroup(nn.Module):
    def __init__(self, channels):
        super(DecoderGroup, self).__init__()
        self._seq = nn.Sequential(DecoderIdentityResBlock(channels, ex=6),
                                  DecoderIdentityResBlock(channels, ex=6))

    def forward(self, x):
        return self._seq(x)


class EncoderScale(nn.Module):
    def __init__(self, in_channels, out_channels, groups, has_fr_res_block=True, save_last_result=False):
        super(EncoderScale, self).__init__()
        self._has_fr_res_block = has_fr_res_block
        self._save_last_result = save_last_result
        self._ml = nn.ModuleList([EncoderGroup(in_channels) for _ in range(groups)])
        if self._has_fr_res_block:
            self._down_enc = FactorizedReduceResBlock(in_channels, out_channels)

    def forward(self, x):
        results = []
        for blah in self._ml:
            x = blah(x)
            results.append(x)
        if self._has_fr_res_block:
            x = self._down_enc(x)
        if not self._save_last_result:
            results.pop()
        return x, results


class Encoder(nn.Module):

    def __init__(self,
                 input_channels=3,
                 num_scales=5,
                 max_groups_per_scale=16,
                 min_groups_per_scale=4,
                 num_latent_per_group=20,
                 num_hidden_channels=30):
        super(Encoder, self).__init__()
        self._seq = nn.Sequential(Stem(input_channels,
                                       num_hidden_channels),
                                  Preprocess(num_hidden_channels,
                                             num_hidden_channels * 2))
        self._encoder_tower = nn.ModuleList()
        for scale_idx in range(num_scales):
            scale_powers_of_two = 2 ** scale_idx
            scale_in_channels = scale_powers_of_two * num_hidden_channels * 2
            scale_out_channels = scale_in_channels * 2
            has_fr_res_block = True
            save_last_result = True
            if scale_idx == num_scales - 1:
                scale_out_channels = scale_in_channels
                has_fr_res_block = False
                save_last_result = False
            self._encoder_tower.append(
                EncoderScale(scale_in_channels,
                             scale_out_channels,
                             max(max_groups_per_scale // scale_powers_of_two, min_groups_per_scale),
                             has_fr_res_block=has_fr_res_block,
                             save_last_result=save_last_result))

        final_encoding_channels = 2 ** (num_scales - 1) * 2 * num_hidden_channels
        self._final_encoding = nn.Sequential(
            nn.ELU(),
            WeightNormedConv2d(final_encoding_channels, final_encoding_channels, kernel_size=1,
                               padding=0, bias=True),
            nn.ELU())
        self._sampler = nn.Sequential(
            WeightNormedConv2d(final_encoding_channels, num_latent_per_group * 2, kernel_size=3,
                               padding=1, bias=True))

    def forward(self, inpt):
        x = self._seq(inpt)
        results = []
        for i, scale in enumerate(self._encoder_tower):
            x, inner_results = scale(x)
            results.append(inner_results)
        results = list(reversed(list(itertools.chain.from_iterable(results))))
        mu, log_var = self._sampler(self._final_encoding(x)).chunk(2, dim=1)

        return mu, log_var, results


class SCombiner(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(SCombiner, self).__init__()
        self.conv = WeightNormedConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, s1, s2):
        s2 = self.conv(s2)
        return s1 + s2


class SZCombiner(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(SZCombiner, self).__init__()
        self.conv = WeightNormedConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, s, z):
        return self.conv(torch.cat([s, z], dim=1))


class Decoder(nn.Module):

    def __init__(self,
                 output_channels=3,
                 output_size=256,
                 spatial_scaling=32,
                 num_scales=5,
                 max_groups_per_scale=16,
                 min_groups_per_scale=4,
                 num_latent_per_group=20,
                 num_hidden_channels=30):
        super(Decoder, self).__init__()
        mult_hidden_channels = num_hidden_channels * 2
        self._num_scales = num_scales
        self._max_groups_per_scale = max_groups_per_scale
        self._min_groups_per_scale = min_groups_per_scale
        self._s_combiners = nn.ModuleList()
        self._s_z_combiners = nn.ModuleList()
        self._groups = nn.ModuleList()
        self._upsamplers = nn.ModuleList()
        self._decoder_samplers = nn.ModuleList()
        self._encoder_samplers = nn.ModuleList()
        for scale_idx in reversed(range(self._num_scales)):
            scale_powers_of_two = 2 ** scale_idx
            scale_channels = scale_powers_of_two * mult_hidden_channels
            num_groups = max(self._max_groups_per_scale // scale_powers_of_two, self._min_groups_per_scale)
            for j in range(num_groups):
                if not ((scale_idx == self._num_scales - 1) and (j == num_groups - 1)):
                    self._s_combiners.append(SCombiner(scale_channels, scale_channels))
                    self._groups.append(DecoderGroup(scale_channels))
                    self._encoder_samplers.append(
                        WeightNormedConv2d(scale_channels, 2 * num_latent_per_group, kernel_size=3,
                                           padding=1, bias=True))
                    self._decoder_samplers.append(nn.Sequential(nn.ELU(),
                                                                WeightNormedConv2d(scale_channels,
                                                                                   2 * num_latent_per_group,
                                                                                   kernel_size=1,
                                                                                   padding=0,
                                                                                   bias=True)))
                self._s_z_combiners.append(SZCombiner(scale_channels + num_latent_per_group, scale_channels))
            if not scale_idx == 0:
                self._upsamplers.append(DecoderUpsample(scale_channels, scale_channels // 2, 1, ex=6))

        self._prior = nn.Parameter(torch.rand(size=spatial_scaling, requires_grad=True))
        self._post_process = Postprocess(in_channels=2 * output_size, out_channels=(2 * output_size) // 2)
        self._image_conditional = nn.Sequential(nn.ELU(),
                                    WeightNormedConv2d((2 * output_size) // 2, output_channels, 3, padding=1, bias=True))

    def forward(self, z, results):
        s = self._prior.unsqueeze(0)
        s = s.expand(z.size(0), -1, -1, -1)
        s = self._s_z_combiners[0](s, z)
        full_idx = 0
        all_kls = []
        for scale_idx in range(self._num_scales):
            scale_powers_of_two = 2 ** (self._num_scales - 1 - scale_idx)
            num_groups = max(self._max_groups_per_scale // scale_powers_of_two,
                             self._min_groups_per_scale)
            num_groups = num_groups - 1 if scale_idx == 0 else num_groups
            for group_idx in range(num_groups):
                # print("group_idx: {} out of: {}".format(group_idx, num_groups))
                # print("full_idx: {}".format(full_idx))
                # print("s: {}".format(torch.isnan(s).any()))
                # print(s.shape)
                # print(self._groups[full_idx])
                s = self._groups[full_idx](s)
                # print("s after groups: {}".format(torch.isnan(s).any()))
                # print(s.shape)
                # print(self._decoder_samplers[full_idx])
                param = self._decoder_samplers[full_idx](s)
                # print("param dec: {}".format(torch.isnan(param).any()))
                mu_p, log_sig_p = torch.chunk(param, 2, dim=1)

                # print(s.shape)
                # print(results[full_idx].shape)
                # print(self._s_combiners[full_idx])
                ftr = self._s_combiners[full_idx](results[full_idx], s)
                # print("ftr: {}".format(torch.isnan(ftr).any()))
                param = self._encoder_samplers[full_idx](ftr)
                # print("param enc: {}".format(torch.isnan(param).any()))

                mu_q, log_sig_q = torch.chunk(param, 2, dim=1)

                mu = mu_p + mu_q
                mu = mu.div(5.).tanh().mul(5.)
                # print("mu: {}".format(torch.isnan(mu).any()))
                mu_p = mu_p.div(5.).tanh().mul(5.)
                # print("mu_p: {}".format(torch.isnan(mu_p).any()))
                log_sig = log_sig_p + log_sig_q
                log_sig = log_sig.div(5.).tanh().mul(5.)
                sig = torch.exp(log_sig) + 1e-2
                # print("log_sig: {}".format(torch.isnan(log_sig).any()))
                log_sig_p = log_sig_p.div(5.).tanh().mul(5.)
                sig_p = torch.exp(log_sig_p) + 1e-2
                # print("log_sig_p: {}".format(torch.isnan(log_sig_p).any()))


                term1 = (mu - mu_p) / sig_p
                # print("term1: {}".format(torch.isnan(term1).any()))

                term2 = sig / sig_p
                # print("term2: {}".format(torch.isnan(term2).any()))

                kl_per_var = 0.5 * (term1 * term1 + term2 * term2) - 0.5 - torch.log(term2)
                # print("large mult: {}".format(torch.isnan(term1 * term1 + term2 * term2).any()))
                # print(term2)
                # print("log: {}".format(torch.isnan(torch.log(term2)).any()))
                # print("kl_per_var: {}".format(torch.isnan(kl_per_var).any()))
                all_kls.append(torch.sum(kl_per_var, dim=[1, 2, 3]))

                z = torch.randn_like(mu) * sig + mu
                # print("z dec: {}".format(torch.isnan(z).any()))
                s = self._s_z_combiners[full_idx + 1](s, z)
                # print("s after s_z combiner: {}".format(torch.isnan(s).any()))
                full_idx += 1
            if not scale_idx == self._num_scales - 1:
                s = self._upsamplers[scale_idx](s)
                # print("s after upsampler: {}".format(torch.isnan(s).any()))

        s = self._post_process(s)

        logits = self._image_conditional(s)

        return logits, all_kls


# class VAE(nn.Module):
#
#     def __init__(self,
#                  input_channels=3,
#                  num_scales=5,
#                  max_groups_per_scale=16,
#                  min_groups_per_scale=4,
#                  num_latent_per_group=20,
#                  num_hidden_channels=30):
#         super(VAE, self).__init__()
#         self._encoder = Encoder()
#         self._decoder = Decoder()
#
#     def forward(self, x):
#         mu, log_var, results = self._encoder(x)
#
#         z = torch.randn_like(mu) * torch.exp(log_var) + mu
#         logits, all_kls = self._decoder(z, results)
#
#         all_kls = torch.stack(all_kls, dim=1)
#         kl = torch.sum(all_kls, dim=1)
#         recon_loss = F.mse_loss(logits, x)
#         loss = kl + recon_loss
#
#         return loss


class VAE(pl.LightningModule):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        # enc_type: str = 'resnet18',
        # first_conv: bool = False,
        # maxpool1: bool = False,
        # enc_out_dim: int = 512,
        # kl_coeff: float = 0.1,
        # latent_dim: int = 256,
        lr: float = 1e-4,
        **kwargs
    ):
        """
        Args:
            input_height: height of the images
            enc_type: option between resnet18 or resnet50
            first_conv: use standard kernel_size 7, stride 2 at start or
                replace it with kernel_size 3, stride 1 conv
            maxpool1: use standard maxpool to reduce spatial dim of feat by a factor of 2
            enc_out_dim: set according to the out_channel count of
                encoder used (512 for resnet18, 2048 for resnet50)
            kl_coeff: coefficient for kl term of the loss
            latent_dim: dim of latent space
            lr: learning rate for Adam
        """

        super(VAE, self).__init__()

        self.save_hyperparameters()
        c_scaling = 2 ** (1 + 2 - 1)
        spatial_scaling = 2 ** (1 + 2 - 1)
        num_scales = 2
        max_groups_per_scale = 10
        min_groups_per_scale = 4
        num_latent_per_group = 20
        num_hidden_channels = 32
        prior_ftr0_size = (
            int(c_scaling * num_hidden_channels), input_shape[1] // spatial_scaling,
            input_shape[1] // spatial_scaling)
        self.lr = lr
        # self.kl_coeff = kl_coeff
        # self.enc_out_dim = enc_out_dim
        # self.latent_dim = latent_dim
        # self.input_height = input_height
        self._encoder = Encoder(input_channels=input_shape[0],
                                num_scales=num_scales,
                                max_groups_per_scale=max_groups_per_scale,
                                min_groups_per_scale=min_groups_per_scale,
                                num_latent_per_group=num_latent_per_group,
                                num_hidden_channels=num_hidden_channels)
        self._decoder = Decoder(output_channels=input_shape[0],
                                output_size=input_shape[1],
                                spatial_scaling=prior_ftr0_size,
                                num_scales=num_scales,
                                max_groups_per_scale=max_groups_per_scale,
                                min_groups_per_scale=min_groups_per_scale,
                                num_latent_per_group=num_latent_per_group,
                                num_hidden_channels=num_hidden_channels)

    def forward(self, x):
        mu, log_var, results = self._encoder(x)
        z = torch.randn_like(mu) * torch.exp(log_var) + mu
        logits, _ = self._decoder(z, results)
        return logits

    def _run_step(self, x):
        mu_q, log_var_q, results = self._encoder(x)

        mu_q = mu_q.div(5.).tanh().mul(5.)
        log_var_q = log_var_q.div(5.).tanh().mul(5.)
        sig = torch.exp(log_var_q) + 1e-2

        z = torch.randn_like(mu_q) * sig + mu_q

        # print("mu: {}".format(torch.isnan(mu).any()))
        # print("log_var: {}".format(torch.isnan(log_var).any()))
        # Need to also calculate first kl here
        mu_p = torch.zeros_like(z)
        sig_p = torch.ones_like(z)
        term1 = (mu_q - mu_p) / sig_p
        term2 = sig / sig_p
        kl_per_var = 0.5 * (term1 * term1 + term2 * term2) - 0.5 - torch.log(term2)
        # print("z: {}".format(torch.isnan(z).any()))
        logits, all_kls = self._decoder(z, results)
        all_kls.insert(0, torch.sum(kl_per_var, dim=[1, 2, 3]))
        # print("logits: {}".format(torch.isnan(logits).any()))
        return logits, all_kls

    def step(self, batch, batch_idx):
        x, y = batch
        logits, all_kls = self._run_step(x)

        all_kls = torch.stack(all_kls, dim=1)
        kl = torch.sum(all_kls, dim=1).mean()
        recon_loss = F.mse_loss(logits, x)
        loss = recon_loss + kl

        logs = {
            "recon_loss": recon_loss,
            "kl": kl,
            "loss": loss,
        }
        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


if __name__ == '__main__':
    pass
    # input_channels = 3
    # num_scales = 5
    # max_groups_per_scale = 16
    # min_groups_per_scale = 4
    # num_latent_per_group = 20
    # num_hidden_channels = 30
    # hidden_channel_mult = 2

    #summary(VAE(input_shape=(3, 32, 32), num_scales=5, max_groups_per_scale=16, min_groups_per_scale=4,
    #                num_latent_per_group=20, num_hidden_channels=30),
    #        input_size=(1, 3, 32, 32), depth=99)
    # summary(DecoderConvSEBlock(in_channels=960, out_channels=960, stride=1), input_size=(1, 960, 8, 8), depth=99)
