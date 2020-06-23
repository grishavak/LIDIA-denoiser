from torch.nn.functional import unfold
from torch.nn.functional import fold
import torch.nn as nn
import torch.nn.functional as nn_func
from torch.nn.functional import conv2d
from utils import *


class ArchitectureOptions:
    def __init__(self, rgb, small_network):
        assert not (rgb and small_network), "LIDIA-S doesn't support color images"
        self.rgb = rgb
        self.small_network = small_network


class VerHorMat(nn.Module):
    def __init__(self, ver_in, ver_out, hor_in, hor_out):
        super(VerHorMat, self).__init__()
        self.ver = nn.Linear(in_features=ver_in, out_features=ver_out, bias=False)
        self.hor = nn.Linear(in_features=hor_in, out_features=hor_out, bias=False)
        self.b = nn.Parameter(torch.empty(hor_out, ver_out, dtype=torch.float32))
        nn.init.xavier_normal_(self.b, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        x = self.ver(x)
        x = self.hor(x.transpose(-1, -2)).transpose(-1, -2)
        x = x + self.b

        return x

    def extra_repr(self):
        return 'b.shape=' + str(tuple(self.b.shape))

    def get_ver_out(self):
        return self.ver.out_features

    def get_hor_out(self):
        return self.hor.out_features


class Aggregation0(nn.Module):
    def __init__(self, patch_w):
        super(Aggregation0, self).__init__()
        self.patch_w = patch_w

    def forward(self, x, pixels_h, pixels_w):
        images, patches, hor_f, ver_f = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(images * hor_f, ver_f, patches)
        patch_cnt = torch.ones(x[0:1, ...].shape, device=x.device)
        patch_cnt = fold(patch_cnt, (pixels_h, pixels_w), (self.patch_w, self.patch_w))
        x = fold(x, (pixels_h, pixels_w), (self.patch_w, self.patch_w)) / patch_cnt
        x = unfold(x, (self.patch_w, self.patch_w))
        x = x.view(images, hor_f, ver_f, patches).permute(0, 3, 1, 2)

        return x


class Aggregation1(nn.Module):
    def __init__(self, patch_w):
        super(Aggregation1, self).__init__()
        self.patch_w = patch_w

        kernel_1d = torch.tensor((1 / 4, 1 / 2, 1 / 4), dtype=torch.float32)
        kernel_2d = (kernel_1d.view(-1, 1) * kernel_1d).view(1, 1, 3, 3)
        self.bilinear_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), bias=False)
        self.bilinear_conv.weight.data = kernel_2d
        self.bilinear_conv.weight.requires_grad = False

    def forward(self, x, pixels_h, pixels_w):
        images, patches, hor_f, ver_f = x.shape
        x = x.permute(0, 2, 3, 1).view(images * hor_f, ver_f, patches)
        patch_cnt = torch.ones(x[0:1, ...].shape, device=x.device)
        patch_cnt = fold(patch_cnt, (pixels_h, pixels_w), (self.patch_w, self.patch_w), dilation=(2, 2))
        x = fold(x, (pixels_h, pixels_w), (self.patch_w, self.patch_w), dilation=(2, 2)) / patch_cnt
        x_b, x_c, x_h, x_w = x.shape
        x = self.bilinear_conv(nn_func.pad(x, [1] * 4, 'reflect').view(x_b * x_c, 1, x_h + 2, x_w + 2)) \
            .view(x_b, x_c, x_h, x_w)
        x = unfold(x, (self.patch_w, self.patch_w), dilation=(2, 2))
        x = x.view(images, hor_f, ver_f, patches).permute(0, 3, 1, 2)

        return x


class VerHorBnRe(nn.Module):
    def __init__(self, ver_in, ver_out, hor_in, hor_out, bn):
        super(VerHorBnRe, self).__init__()
        self.ver_hor = VerHorMat(ver_in=ver_in, ver_out=ver_out, hor_in=hor_in, hor_out=hor_out)
        if bn:
            self.bn = nn.BatchNorm2d(hor_out)
        self.thr = nn.ReLU()

    def forward(self, x):
        x = self.ver_hor(x)
        if hasattr(self, 'bn'):
            x = self.bn(x.transpose(-2, -3)).transpose(-2, -3)
        x = self.thr(x)

        return x


class SeparablePart1(nn.Module):
    def __init__(self, arch_opt, hor_size, patch_numel, ver_size):
        super(SeparablePart1, self).__init__()

        self.ver_hor_bn_re0 = VerHorBnRe(ver_in=patch_numel, ver_out=ver_size, hor_in=14, hor_out=hor_size, bn=False)
        if not arch_opt.small_network:
            self.a = nn.Parameter(torch.tensor((0,), dtype=torch.float32))
            self.ver_hor_bn_re1 = VerHorBnRe(ver_in=ver_size, ver_out=ver_size, hor_in=14, hor_out=hor_size, bn=True)

    def forward(self, x):
        x = self.ver_hor_bn_re0(x)
        if hasattr(self, 'ver_hor_bn_re1'):
            x = self.a * x + (1 - self.a) * self.ver_hor_bn_re1(x)

        return x


class SeparablePart2(nn.Module):
    def __init__(self, arch_opt, hor_size_in, patch_numel, ver_size):
        super(SeparablePart2, self).__init__()

        self.ver_hor_bn_re2 = VerHorBnRe(ver_in=ver_size, ver_out=ver_size, hor_in=hor_size_in, hor_out=56, bn=True)
        if not arch_opt.small_network:
            self.a = nn.Parameter(torch.tensor((0,), dtype=torch.float32))
            self.ver_hor_bn_re3 = VerHorBnRe(ver_in=ver_size, ver_out=ver_size, hor_in=56, hor_out=56, bn=True)
        self.ver_hor_out = VerHorMat(ver_in=ver_size, ver_out=patch_numel, hor_in=56, hor_out=1)

    def forward(self, x):
        x = self.ver_hor_bn_re2(x)
        if hasattr(self, 'ver_hor_bn_re3'):
            x = self.a * x + (1 - self.a) * self.ver_hor_bn_re3(x)
        x = self.ver_hor_out(x)

        return x


class SeparableFcNet(nn.Module):
    def __init__(self, arch_opt, patch_w, ver_size):
        super(SeparableFcNet, self).__init__()
        patch_numel = (patch_w ** 2) * 3 if arch_opt.rgb else patch_w ** 2

        self.sep_part1_s0 = SeparablePart1(arch_opt=arch_opt, hor_size=14, patch_numel=patch_numel, ver_size=ver_size)
        self.sep_part1_s1 = SeparablePart1(arch_opt=arch_opt, hor_size=14, patch_numel=patch_numel, ver_size=ver_size)
        self.ver_hor_agg0_pre = VerHorMat(ver_in=ver_size, ver_out=patch_numel, hor_in=14, hor_out=1)
        self.agg0 = Aggregation0(patch_w)
        self.ver_hor_bn_re_agg0_post = VerHorBnRe(ver_in=patch_numel, ver_out=ver_size, hor_in=1, hor_out=14, bn=False)
        self.ver_hor_agg1_pre = VerHorMat(ver_in=ver_size, ver_out=patch_numel, hor_in=14, hor_out=1)
        self.agg1 = Aggregation1(patch_w)
        self.ver_hor_bn_re_agg1_post = VerHorBnRe(ver_in=patch_numel, ver_out=ver_size, hor_in=1, hor_out=14, bn=False)

        self.sep_part2 = SeparablePart2(arch_opt=arch_opt, hor_size_in=56, patch_numel=patch_numel, ver_size=ver_size)

    def forward(self, x0, x1, weights1, im_params0, im_params1, save_memory, max_chunk):
        if save_memory:
            out = torch.zeros(x0[:, :, 0:1, :].shape, device=x0.device).fill_(float('nan'))
            out_s0_shape = torch.Size((x0.shape[0:2]) +
                                      (self.sep_part1_s0.ver_hor_bn_re0.ver_hor.get_hor_out(),) +
                                      (self.sep_part1_s0.ver_hor_bn_re0.ver_hor.get_ver_out(),))
            out_part1_s0 = torch.zeros(out_s0_shape, device=x0.device).fill_(float('nan'))
            apply_on_chuncks(self.sep_part1_s0, x0, max_chunk, out_part1_s0)

            # sep_part1_s1
            out_s1_shape = torch.Size((x1.shape[0:2]) +
                                      (self.sep_part1_s1.ver_hor_bn_re0.ver_hor.get_hor_out(),) +
                                      (self.sep_part1_s1.ver_hor_bn_re0.ver_hor.get_ver_out(),))
            out_part1_s1 = torch.zeros(out_s1_shape, device=x1.device).fill_(float('nan'))
            apply_on_chuncks(self.sep_part1_s1, x1, max_chunk, out_part1_s1)

            # agg0
            y0_shape = torch.Size((out_part1_s0[:, :, 0:1, :].shape[0:3]) +
                                  (self.ver_hor_agg0_pre.get_ver_out(),))
            y_tmp0 = torch.zeros(y0_shape, device=out_part1_s0.device).fill_(float('nan'))
            apply_on_chuncks(self.ver_hor_agg0_pre, out_part1_s0, max_chunk, y_tmp0)
            y_tmp0 = self.agg0(y_tmp0, im_params0['pixels_h'], im_params0['pixels_w'])

            # agg1
            y1_shape = torch.Size((out_part1_s1[:, :, 0:1, :].shape[0:3]) +
                                  (self.ver_hor_agg1_pre.get_ver_out(),))
            y_tmp1 = torch.zeros(y1_shape, device=out_part1_s1.device).fill_(float('nan'))
            apply_on_chuncks(self.ver_hor_agg1_pre, out_part1_s1, max_chunk, y_tmp1)
            y_tmp1 = weights1 * self.agg1(y_tmp1 / weights1, im_params1['pixels_h'], im_params1['pixels_w'])

            # sep_part2
            for si in range(0, out_part1_s0.shape[1], max_chunk):
                ei = min(si + max_chunk, out_part1_s0.shape[1])
                y_out0 = self.ver_hor_bn_re_agg0_post(y_tmp0[:, si:ei, :, :])
                y_out1 = self.ver_hor_bn_re_agg1_post(y_tmp1[:, si:ei, :, :])
                out[:, si:ei, :, :] = self.sep_part2(torch.cat((out_part1_s0[:, si:ei, :, :],
                                                                out_part1_s1[:, si:ei, :, :],
                                                                y_out0, y_out1), dim=-2))

        else:
            # sep_part1
            x0 = self.sep_part1_s0(x0)
            x1 = self.sep_part1_s1(x1)

            # agg0
            y_out0 = self.ver_hor_agg0_pre(x0)
            y_out0 = self.agg0(y_out0, im_params0['pixels_h'], im_params0['pixels_w'])
            y_out0 = self.ver_hor_bn_re_agg0_post(y_out0)

            # agg1
            y_out1 = self.ver_hor_agg1_pre(x1)
            y_out1 = weights1 * self.agg1(y_out1 / weights1, im_params1['pixels_h'], im_params1['pixels_w'])
            y_out1 = self.ver_hor_bn_re_agg1_post(y_out1)

            # sep_part2
            out = self.sep_part2(torch.cat((x0, x1, y_out0, y_out1), dim=-2))

        return out


class FcNet(nn.Module):
    def __init__(self):
        super(FcNet, self).__init__()
        for layer in range(6):
            self.add_module('fc{}'.format(layer), nn.Linear(in_features=14,
                                                            out_features=14,
                                                            bias=False))
            self.add_module('bn{}'.format(layer), nn.BatchNorm1d(14))
            self.add_module('relu{}'.format(layer), nn.ReLU())

        self.add_module('fc_out', nn.Linear(in_features=14, out_features=14, bias=True))

    def forward(self, x):
        for name, module in self._modules.items():
            if 'bn' in name:
                images, patches, values = x.shape
                x = module(x.view(images * patches, values)).view(images, patches, values)
            else:
                x = module(x)
        return x


class PatchDenoiseNet(nn.Module):
    def __init__(self, arch_opt, patch_w, ver_size):
        super(PatchDenoiseNet, self).__init__()
        self.arch_opt = arch_opt
        self.separable_fc_net = SeparableFcNet(arch_opt=arch_opt, patch_w=patch_w, ver_size=ver_size)
        self.weights_net0 = FcNet()
        self.alpha0 = nn.Parameter(torch.tensor((0.5,), dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor((0.5,), dtype=torch.float32))

        self.weights_net1 = FcNet()
        self.alpha1 = nn.Parameter(torch.tensor((0.5,), dtype=torch.float32))

    def forward(self, patches_n0, dist0, patches_n1, dist1, im_params0, im_params1, save_memory, max_chunk):

        weights0 = self.weights_net0(torch.exp(-self.alpha0.abs() * dist0)).unsqueeze(-1)
        weighted_patches_n0 = patches_n0 * weights0

        weights1 = self.weights_net1(torch.exp(-self.alpha1.abs() * dist1)).unsqueeze(-1)
        weighted_patches_n1 = patches_n1 * weights1
        weights1_first = weights1[:, :, 0:1, :]

        noise = self.separable_fc_net(weighted_patches_n0, weighted_patches_n1, weights1_first,
                                      im_params0, im_params1, save_memory, max_chunk)

        patches_dn = patches_n0[:, :, 0, :] - noise.squeeze(-2)
        patches_no_mean = patches_dn - patches_dn.mean(dim=-1, keepdim=True)
        patch_exp_weights = (patches_no_mean ** 2).mean(dim=-1, keepdim=True)
        patch_weights = torch.exp(-self.beta.abs() * patch_exp_weights)

        return patches_dn, patch_weights


class NonLocalDenoiser(nn.Module):
    def __init__(self, pad_offs, arch_opt):
        super(NonLocalDenoiser, self).__init__()
        self.arch_opt = arch_opt
        self.pad_offs = pad_offs
        self.patch_w = 5 if arch_opt.rgb else 7
        self.ver_size = 80 if arch_opt.rgb else 64

        self.rgb2gray = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=(1, 1), bias=False)
        self.rgb2gray.weight.data = torch.tensor([0.2989, 0.5870, 0.1140], dtype=torch.float32).view(1, 3, 1, 1)
        self.rgb2gray.weight.requires_grad = False

        kernel_1d = torch.tensor((1 / 4, 1 / 2, 1 / 4), dtype=torch.float32)
        kernel_2d = (kernel_1d.view(-1, 1) * kernel_1d).view(1, 1, 3, 3)
        self.bilinear_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), bias=False)
        self.bilinear_conv.weight.data = kernel_2d
        self.bilinear_conv.weight.requires_grad = False

        self.patch_denoise_net = PatchDenoiseNet(arch_opt=arch_opt, patch_w=self.patch_w, ver_size=self.ver_size)

    def find_nn(self, image_pad, im_params, patch_w, scale=0, case=None):
        neigh_patches_w = 2 * 14 + 1
        top_dist = torch.zeros(im_params['batches'], im_params['patches_h'],
                               im_params['patches_w'], neigh_patches_w ** 2,
                               device=image_pad.device).fill_(float('nan'))
        dist_filter = torch.ones(1, 1, patch_w, patch_w, device=image_pad.device)

        y = image_pad[:, :, 14:14 + im_params['pixels_h'], 14:14 + im_params['pixels_w']]
        for row in range(neigh_patches_w):
            for col in range(neigh_patches_w):
                lin_ind = row * neigh_patches_w + col
                y_shift = image_pad[:, :,
                                    row:row + im_params['pixels_h'],
                                    col:col + im_params['pixels_w']]
                top_dist[:, :, :, lin_ind] = conv2d((y_shift - y) ** 2, dist_filter).squeeze(1)

        top_dist, top_ind = torch.topk(top_dist, 14, dim=3, largest=False, sorted=True)
        top_ind_rows = top_ind // neigh_patches_w
        top_ind_cols = top_ind % neigh_patches_w
        col_arange = torch.arange(im_params['patches_w'], device=image_pad.device).view(1, 1, -1, 1)
        row_arange = torch.arange(im_params['patches_h'], device=image_pad.device).view(1, -1, 1, 1)
        if scale == 1:
            if case == '00':
                top_ind_rows = top_ind_rows * 2 + row_arange * 2
                top_ind_cols = top_ind_cols * 2 + col_arange * 2
            elif case == '10':
                top_ind_rows = top_ind_rows * 2 + 1 + row_arange * 2
                top_ind_cols = top_ind_cols * 2 + col_arange * 2
            elif case == '01':
                top_ind_rows = top_ind_rows * 2 + row_arange * 2
                top_ind_cols = top_ind_cols * 2 + 1 + col_arange * 2
            elif case == '11':
                top_ind_rows = top_ind_rows * 2 + 1 + row_arange * 2
                top_ind_cols = top_ind_cols * 2 + 1 + col_arange * 2
            else:
                assert False
        elif scale == 0:
            top_ind_rows += row_arange
            top_ind_cols += col_arange
        else:
            assert False
        top_ind = top_ind_rows * im_params['pad_patches_w_full'] + top_ind_cols

        return top_dist, top_ind

    def denoise_image(self, image_n, train, save_memory, max_batch):

        patch_numel = (self.patch_w ** 2) * image_n.shape[1]

        image_n0 = self.pad_crop0(image_n, self.pad_offs, train)
        if self.arch_opt.rgb:
            image_for_nn0 = self.rgb2gray(image_n0)
        else:
            image_for_nn0 = image_n0

        im_params0 = get_image_params(image_n0, self.patch_w, 14)
        im_params0['pad_patches_w_full'] = im_params0['pad_patches_w']
        top_dist0, top_ind0 = self.find_nn(image_for_nn0, im_params0, self.patch_w)
        top_ind0 += im_params0['pad_patches'] * torch.arange(top_ind0.shape[0],
                                                             device=image_n0.device).view(-1, 1, 1, 1)
        patch_dist0 = top_dist0.view(top_dist0.shape[0], -1, 14)[:, :, 1:]

        im_patches_n0 = unfold(image_n0, (self.patch_w, self.patch_w)).transpose(1, 0)\
            .contiguous().view(patch_numel, -1).t()
        im_patches_n0 = im_patches_n0[top_ind0.view(-1), :].view(top_ind0.shape[0], -1, 14, patch_numel)

        patch_var0 = im_patches_n0[:, :, 0, :].std(dim=-1).unsqueeze(-1).pow(2) * patch_numel
        patch_dist0 = torch.cat((patch_dist0, patch_var0), dim=-1)

        image_n1 = self.pad_crop1(image_n, train, 'reflect')
        im_n1_b, im_n1_c, im_n1_h, im_n1_w = image_n1.shape
        image_n1 = self.bilinear_conv(image_n1.view(im_n1_b * im_n1_c, 1, im_n1_h, im_n1_w))\
            .view(im_n1_b, im_n1_c, im_n1_h - 2, im_n1_w - 2)
        image_n1 = self.pad_crop1(image_n1, train, 'constant')

        im_params1 = get_image_params(image_n1, 2 * self.patch_w - 1, 28)
        im_params1['pad_patches_w_full'] = im_params1['pad_patches_w']

        if self.arch_opt.rgb:
            image_for_nn1 = self.rgb2gray(image_n1)
        else:
            image_for_nn1 = image_n1

        image_for_nn1_00 = image_for_nn1[:, :, 0::2, 0::2].clone()
        image_for_nn1_10 = image_for_nn1[:, :, 1::2, 0::2].clone()
        image_for_nn1_01 = image_for_nn1[:, :, 0::2, 1::2].clone()
        image_for_nn1_11 = image_for_nn1[:, :, 1::2, 1::2].clone()

        im_params1_00 = get_image_params(image_for_nn1_00, self.patch_w, 14)
        im_params1_10 = get_image_params(image_for_nn1_10, self.patch_w, 14)
        im_params1_01 = get_image_params(image_for_nn1_01, self.patch_w, 14)
        im_params1_11 = get_image_params(image_for_nn1_11, self.patch_w, 14)
        im_params1_00['pad_patches_w_full'] = im_params1['pad_patches_w']
        im_params1_10['pad_patches_w_full'] = im_params1['pad_patches_w']
        im_params1_01['pad_patches_w_full'] = im_params1['pad_patches_w']
        im_params1_11['pad_patches_w_full'] = im_params1['pad_patches_w']
        top_dist1_00, top_ind1_00 = self.find_nn(image_for_nn1_00, im_params1_00, self.patch_w, scale=1, case='00')
        top_dist1_10, top_ind1_10 = self.find_nn(image_for_nn1_10, im_params1_10, self.patch_w, scale=1, case='10')
        top_dist1_01, top_ind1_01 = self.find_nn(image_for_nn1_01, im_params1_01, self.patch_w, scale=1, case='01')
        top_dist1_11, top_ind1_11 = self.find_nn(image_for_nn1_11, im_params1_11, self.patch_w, scale=1, case='11')

        top_dist1 = torch.zeros(im_params1['batches'], im_params1['patches_h'],
                                im_params1['patches_w'], 14,
                                device=image_n.device).fill_(float('nan'))
        top_dist1[:, 0::2, 0::2, :] = top_dist1_00
        top_dist1[:, 1::2, 0::2, :] = top_dist1_10
        top_dist1[:, 0::2, 1::2, :] = top_dist1_01
        top_dist1[:, 1::2, 1::2, :] = top_dist1_11

        top_ind1 = im_params1['pad_patches'] * torch.ones(top_dist1.shape, dtype=torch.int64, device=image_n.device)
        top_ind1[:, 0::2, 0::2, :] = top_ind1_00
        top_ind1[:, 1::2, 0::2, :] = top_ind1_10
        top_ind1[:, 0::2, 1::2, :] = top_ind1_01
        top_ind1[:, 1::2, 1::2, :] = top_ind1_11

        top_ind1 += im_params1['pad_patches'] * torch.arange(top_ind1.shape[0],
                                                             device=image_n1.device).view(-1, 1, 1, 1)

        im_patches_n1 = unfold(image_n1, (self.patch_w, self.patch_w),
                               dilation=(2, 2)).transpose(1, 0).contiguous().view(patch_numel, -1).t()
        im_patches_n1 = im_patches_n1[top_ind1.view(-1), :].view(top_ind1.shape[0], -1, 14, patch_numel)
        patch_dist1 = top_dist1.view(top_dist1.shape[0], -1, 14)[:, :, 1:]
        patch_var1 = im_patches_n1[:, :, 0, :].std(dim=-1).unsqueeze(-1).pow(2) * patch_numel
        patch_dist1 = torch.cat((patch_dist1, patch_var1), dim=-1)

        image_dn, patch_weights = self.patch_denoise_net(im_patches_n0, patch_dist0, im_patches_n1, patch_dist1,
                                                         im_params0, im_params1, save_memory, max_batch)
        image_dn = image_dn * patch_weights
        ones_tmp = torch.ones(1, 1, patch_numel, device=im_patches_n0.device)
        patch_weights = (patch_weights * ones_tmp).transpose(2, 1)
        image_dn = image_dn.transpose(2, 1)
        image_dn = fold(image_dn, (im_params0['pixels_h'], im_params0['pixels_w']),
                        (self.patch_w, self.patch_w))
        patch_cnt = fold(patch_weights, (im_params0['pixels_h'], im_params0['pixels_w']),
                         (self.patch_w, self.patch_w))

        row_offs = min(self.patch_w - 1, im_params0['patches_h'] - 1)
        col_offs = min(self.patch_w - 1, im_params0['patches_w'] - 1)
        image_dn = crop_offset(image_dn, (row_offs,), (col_offs,)) / crop_offset(patch_cnt, (row_offs,), (col_offs,))

        return image_dn

    def pad_crop0(self, image, pad_offs, train):
        if not train:
            reflect_pad = [self.patch_w - 1] * 4
            constant_pad = [14] * 4
            image = nn_func.pad(nn_func.pad(image, reflect_pad, 'reflect'), constant_pad, 'constant', -1)
        else:
            image = crop_offset(image, (pad_offs,), (pad_offs,))

        return image

    def pad_crop1(self, image, train, mode):
        if not train:
            if mode == 'reflect':
                bilinear_pad = 1
                averaging_pad = (self.patch_w - 1) // 2
                patch_w_scale_1 = 2 * self.patch_w - 1
                find_nn_pad = (patch_w_scale_1 - 1) // 2
                reflect_pad = [averaging_pad + bilinear_pad + find_nn_pad] * 4
                image = nn_func.pad(image, reflect_pad, 'reflect')
            elif mode == 'constant':
                constant_pad = [28] * 4
                image = nn_func.pad(image, constant_pad, 'constant', -1)
            else:
                assert False

        return image

    def forward(self, image_n, train=True, save_memory=False, max_chunk=None):
        image_n_mean = image_n.mean(dim=(-2, -1), keepdim=True)
        image_n = image_n - image_n_mean
        image_dn = self.denoise_image(image_n, train, save_memory, max_chunk)
        image_dn = image_dn + image_n_mean

        return image_dn


class Logger(object):
    def __init__(self, fname="logfile.log"):
        self.terminal = sys.stdout
        self.log = open(fname, "w")

    def __del__(self):
        self.log.close()

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass
