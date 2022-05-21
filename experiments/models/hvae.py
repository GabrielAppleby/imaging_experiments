import itertools
from typing import Tuple, Optional, Callable

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import nn as nn
from torch.nn import Identity
# from torchinfo import summary
from torch.optim import Optimizer

from models.discretized_logistic_mixture import DiscMixLogistic
from models.utils import calc_kl_and_z, kl_coefficients


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


class Encoder(nn.Module):

    def __init__(self,
                 input_channels=3,
                 num_scales=5,
                 max_groups_per_scale=16,
                 min_groups_per_scale=4,
                 num_latent_per_group=20,
                 num_hidden_channels=30,
                 device='cuda:0'):
        super(Encoder, self).__init__()
        self.device = device
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

        final_encoding_channels = 2 ** num_scales * num_hidden_channels
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


class Decoder(nn.Module):

    def __init__(self,
                 output_channels=100,
                 final_encoding_size=(96, 8, 8),
                 num_scales=5,
                 max_groups_per_scale=16,
                 min_groups_per_scale=4,
                 num_latent_per_group=20,
                 num_hidden_channels=30,
                 device='none'):
        super(Decoder, self).__init__()
        mult_hidden_channels = num_hidden_channels * 2
        self.device = device
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
        self._prior = nn.Parameter(torch.rand(size=final_encoding_size, requires_grad=True, device=self.device))
        self._post_process = Postprocess(in_channels=2 * num_hidden_channels, out_channels=num_hidden_channels)
        self._image_conditional = nn.Sequential(nn.ELU(),
                                    WeightNormedConv2d(num_hidden_channels, output_channels, 3, padding=1, bias=True))

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

                kl_per_var, z = calc_kl_and_z(mu_p + mu_q, log_sig_p + log_sig_q, mu_q, log_sig_q)
                all_kls.append(kl_per_var)

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


class VAE(pl.LightningModule):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        output_channels: int = 100,
        num_scales: int = 5,
        max_groups_per_scale: int = 16,
        min_groups_per_scale: int = 4,
        num_latent_per_group: int = 20,
        num_hidden_channels: int = 24,
        learning_rate: float = 1e-2,
        min_kl_coeff: float = 0.0001,
        anneal_kl_instances: int = 2700000,
        const_kl_instances: int = 27000,
        **kwargs
    ):
        super(VAE, self).__init__()
        self._input_shape = input_shape
        self._num_scales = num_scales
        self._max_groups_per_scale = max_groups_per_scale
        self._min_groups_per_scale = min_groups_per_scale
        self._num_latent_per_group = num_latent_per_group
        self._num_hidden_channels = num_hidden_channels
        self._min_kl_coeff = min_kl_coeff
        self._anneal_kl_instances = anneal_kl_instances
        self._const_kl_instances = const_kl_instances
        self.learning_rate = learning_rate

        self.save_hyperparameters()
        final_scale = 2 ** self._num_scales
        final_encoding_size = (final_scale * self._num_hidden_channels,
                               self._input_shape[1] // final_scale,
                               self._input_shape[1] // final_scale)
        self._encoder = Encoder(input_channels=self._input_shape[0],
                                num_scales=self._num_scales,
                                max_groups_per_scale=self._max_groups_per_scale,
                                min_groups_per_scale=self._min_groups_per_scale,
                                num_latent_per_group=self._num_latent_per_group,
                                num_hidden_channels=self._num_hidden_channels,
                                device=self.device)
        self._decoder = Decoder(output_channels=output_channels,
                                final_encoding_size=final_encoding_size,
                                num_scales=self._num_scales,
                                max_groups_per_scale=self._max_groups_per_scale,
                                min_groups_per_scale=self._min_groups_per_scale,
                                num_latent_per_group=self._num_latent_per_group,
                                num_hidden_channels=self._num_hidden_channels,
                                device=self.device)
        self._kl_coeffs = kl_coefficients(self._num_scales,
                                          self._max_groups_per_scale,
                                          self._min_groups_per_scale,
                                          device=self.device)

    def forward(self, x):
        mu, log_var, results = self._encoder(x)
        z = torch.randn_like(mu, device=self.device) * torch.exp(log_var) + mu
        logits, _ = self._decoder(z, results)
        return logits

    def _run_step(self, x):
        mu_q, log_sig_q, results = self._encoder(x)

        # print("mu: {}".format(torch.isnan(mu).any()))
        # print("log_var: {}".format(torch.isnan(log_var).any()))
        # Need to also calculate first kl here
        mu_p = torch.zeros_like(mu_q, device=self.device)
        log_sig_p = torch.zeros_like(mu_q, device=self.device)
        kl_per_var, z = calc_kl_and_z(mu_p, log_sig_p, mu_q, log_sig_q)
        # print("z: {}".format(torch.isnan(z).any()))
        logits, all_kls = self._decoder(z, results)
        all_kls.insert(0, kl_per_var)
        # print("logits: {}".format(torch.isnan(logits).any()))
        return logits, all_kls

    def step(self, batch, batch_idx):
        kl_coeff = max(min((self.global_step * len(batch) - self._const_kl_instances) / self._anneal_kl_instances, 1), self._min_kl_coeff)

        x, y = batch
        logits, all_kls = self._run_step(x)

        all_kls = torch.stack(all_kls, dim=1)
        kl_coeff_i = torch.abs(all_kls)
        kl_coeff_i = torch.mean(kl_coeff_i, dim=0, keepdim=True) + 0.01

        total_kl = torch.sum(kl_coeff_i)

        kl_coeff_i = kl_coeff_i / self._kl_coeffs * total_kl
        kl_coeff_i = kl_coeff_i / torch.mean(kl_coeff_i, dim=1, keepdim=True)
        kl = torch.sum(all_kls * kl_coeff_i, dim=1) * kl_coeff

        recon_loss = - torch.sum(DiscMixLogistic(logits, self.device).log_prob(x), dim=[1, 2])

        loss = torch.mean(recon_loss + kl)

        logs = {
            "recon_loss": recon_loss.mean(),
            "kl": kl.mean(),
            "loss": loss,
            "kl_coeff": kl_coeff,
            "lr": self.hparams.learning_rate
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
        optimizer = torch.optim.Adamax(self.parameters(), 1e-2, weight_decay=3e-4, eps=1e-3)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 300, eta_min=1e-4)

        return [optimizer], [scheduler]


if __name__ == '__main__':
    pass
    # input_channels = 3
    # num_scales = 5
    # max_groups_per_scale = 16
    # min_groups_per_scale = 4
    # num_latent_per_group = 20
    # num_hidden_channels = 30
    # hidden_channel_mult = 2
    #
    # summary(VAE(input_shape=(3, 256, 256), num_scales=5, max_groups_per_scale=16, min_groups_per_scale=4,
    #                num_latent_per_group=20, num_hidden_channels=30),
    #        input_size=(1, 3, 256, 256), depth=99)
