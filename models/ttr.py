import torch
import torch.nn as nn
import torch.nn.functional as F


def trilinear(img, lut):
    img = (img - .5) * 2.
    lut = lut[None]
    if img.shape[0] != 1:
        lut = lut.expand(img.shape[0], -1, -1, -1, -1)
    img = img.permute(0, 2, 3, 1)[:, None]
    result = F.grid_sample(lut, img, mode='bilinear', padding_mode='border', align_corners=True)
    return result.squeeze(2)


class LUT3D(nn.Module):
    def __init__(self, dim=33, mode='zero'):
        super(LUT3D, self).__init__()
        if mode == 'zero':
            self.LUT = torch.zeros(3, dim, dim, dim, dtype=torch.float)
            self.LUT = nn.Parameter(self.LUT.clone().detach())
        elif mode == 'identity':
            if dim == 33:
                file = open("./IdentityLUT33.txt", 'r')
            elif dim == 64:
                file = open("./IdentityLUT64.txt", 'r')
            lut = file.readlines()
            self.LUT = torch.zeros(3, dim, dim, dim, dtype=torch.float)
            for i in range(0, dim):
                for j in range(0, dim):
                    for k in range(0, dim):
                        n = i * dim * dim + j * dim + k
                        x = lut[n].split()
                        self.LUT[0, i, j, k] = float(x[0])
                        self.LUT[1, i, j, k] = float(x[1])
                        self.LUT[2, i, j, k] = float(x[2])
            self.LUT = nn.Parameter(self.LUT.clone().detach())
        else:
            raise NotImplementedError

    def forward(self, img):
        return trilinear(img, self.LUT)


def discriminator_block(in_filters, out_filters, normalization=False):
    layers = [nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1), nn.LeakyReLU(0.2)]
    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters, affine=True))
    return layers


class Classifier(nn.Module):
    def __init__(self, in_channels=3, num_class=3):
        super(Classifier, self).__init__()

        self.model = nn.Sequential(
            nn.Upsample(size=(256, 256), mode='bilinear'),
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(16, affine=True),
            *discriminator_block(16, 32, normalization=True),
            *discriminator_block(32, 64, normalization=True),
            *discriminator_block(64, 128, normalization=True),
            *discriminator_block(128, 128),
            # *discriminator_block(128, 128, normalization=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(128, num_class, 8, padding=0),
        )

    def forward(self, img_input):
        return self.model(img_input)


class TTR(nn.Module):
    def __init__(self):
        super(TTR, self).__init__()
        self.lut0 = LUT3D(mode='identity')
        self.lut1 = LUT3D(mode='zero')
        self.lut2 = LUT3D(mode='zero')
        self.classifier = Classifier()

    def forward(self, inp):
        if self.training:
            pred = self.classifier(inp).squeeze()
            if len(pred.shape) == 1:
                pred = pred.unsqueeze(0)
            gen_0 = self.lut0(inp)
            gen_1 = self.lut1(inp)
            gen_2 = self.lut2(inp)
            res = inp.new(inp.size())
            for b in range(inp.size(0)):
                res[b, :, :, :] = pred[b, 0] * gen_0[b, :, :, :] + pred[b, 1] * gen_1[b, :, :, :] + pred[
                    b, 2] * gen_2[b, :, :, :]
            return res
        else:
            res = []
            for i in range(inp.shape[0]):
                pred = self.classifier(inp[i].unsqueeze(0)).squeeze()
                lut = pred[0] * self.lut0.LUT + pred[1] * self.lut1.LUT + pred[2] * self.lut2.LUT
                res.append(trilinear(inp[i].unsqueeze(0), lut))
            res = torch.cat(res)
            return res


if __name__ == '__main__':
    # input_features: shape [B, num_channels, depth, height, width]
    # sampling_grid: shape  [B, depth, height, 3]
    data = torch.randn(2, 3, 512, 512).cuda()
    model = TTR().cuda()
    out = model(data)
    print(out.shape)


# def lut_transform(imgs, luts):
#     # img (b, 3, h, w), lut (b, c, m, m, m)
#
#     # normalize pixel values
#     imgs = (imgs - .5) * 2.
#     # reshape img to grid of shape (b, 1, h, w, 3)
#     grids = imgs.permute(0, 2, 3, 1).unsqueeze(1)
#
#     # after gridsampling, output is of shape (b, c, 1, h, w)
#     outs = F.grid_sample(luts, grids,
#                          mode='bilinear', padding_mode='border', align_corners=True)
#     # remove the extra dimension
#     outs = outs.squeeze(2)
#     return outs
#
#
# class LUT1DGenerator(nn.Module):
#     r"""The 1DLUT generator module.
#     Args:
#         n_colors (int): Number of input color channels.
#         n_vertices (int): Number of sampling points.
#         n_feats (int): Dimension of the input image representation vector.
#         color_share (bool, optional): Whether to share a single 1D LUT across
#             three color channels. Default: False.
#     """
#
#     def __init__(self, n_colors, n_vertices, n_feats, color_share=False) -> None:
#         super().__init__()
#         repeat_factor = n_colors if not color_share else 1
#         self.lut1d_generator = nn.Linear(
#             n_feats, n_vertices * repeat_factor)
#
#         self.n_colors = n_colors
#         self.n_vertices = n_vertices
#         self.color_share = color_share
#
#     def forward(self, x):
#         x = x.view(x.shape[0], -1)
#         lut1d = self.lut1d_generator(x).view(
#             x.shape[0], -1, self.n_vertices)
#         if self.color_share:
#             lut1d = lut1d.repeat_interleave(self.n_colors, dim=1)
#         lut1d = lut1d.sigmoid()
#         return lut1d
#
#
# class LUT3DGenerator(nn.Module):
#     r"""The 3DLUT generator module.
#     Args:
#         n_colors (int): Number of input color channels.
#         n_vertices (int): Number of sampling points along each lattice dimension.
#         n_feats (int): Dimension of the input image representation vector.
#         n_ranks (int): Number of ranks (or the number of basis LUTs).
#     """
#
#     def __init__(self, n_colors, n_vertices, n_feats, n_ranks) -> None:
#         super().__init__()
#
#         # h0
#         self.weights_generator = nn.Linear(n_feats, n_ranks)
#         # h1
#         self.basis_luts_bank = nn.Linear(
#             n_ranks, n_colors * (n_vertices ** n_colors), bias=False)
#
#         self.n_colors = n_colors
#         self.n_vertices = n_vertices
#         self.n_feats = n_feats
#         self.n_ranks = n_ranks
#
#     def init_weights(self):
#         r"""Init weights for models.
#         For the mapping f (`backbone`) and h (`lut_generator`), we follow the initialization in
#             [TPAMI 3D-LUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT).
#         """
#         nn.init.ones_(self.weights_generator.bias)
#         identity_lut = torch.stack([
#             torch.stack(
#                 torch.meshgrid(*[torch.arange(self.n_vertices) for _ in range(self.n_colors)]),
#                 dim=0).div(self.n_vertices - 1).flip(0),
#             *[torch.zeros(
#                 self.n_colors, *((self.n_vertices,) * self.n_colors)) for _ in range(self.n_ranks - 1)]
#         ], dim=0).view(self.n_ranks, -1)
#         self.basis_luts_bank.weight.data.copy_(identity_lut.t())
#
#     def forward(self, x):
#         weights = self.weights_generator(x)
#         luts = self.basis_luts_bank(weights)
#         luts = luts.view(x.shape[0], -1, *((self.n_vertices,) * self.n_colors))
#         return weights, luts
#
#     def regularizations(self, smoothness, monotonicity):
#         basis_luts = self.basis_luts_bank.weight.t().view(
#             self.n_ranks, self.n_colors, *((self.n_vertices,) * self.n_colors))
#         tv, mn = 0, 0
#         for i in range(2, basis_luts.ndimension()):
#             diff = torch.diff(basis_luts.flip(i), dim=i)
#             tv += torch.square(diff).sum(0).mean()
#             mn += F.relu(diff).sum(0).mean()
#         reg_smoothness = smoothness * tv
#         reg_monotonicity = monotonicity * mn
#         return reg_smoothness, reg_monotonicity
#
#
# class LUTNet(nn.Module):
#     # backbone is 'light' or 'res18'
#     def __init__(self, backbone='light'):
#         super(LUTNet, self).__init__()
#         self.backbone = dict(
#             light=LightBackbone,
#             res18=Res18Backbone)[backbone.lower()](
#             pretrained=True,
#             extra_pooling=True,
#             n_base_feats=8)
#         n_colors = 3
#         n_vertices_3d = 17
#         n_vertices_1d = 17
#         n_ranks = 3
#         lut1d_color_share = False
#
#         self.lut3d_generator = LUT3DGenerator(
#             n_colors, n_vertices_3d, self.backbone.out_channels, n_ranks)
#         self.lut1d_generator = LUT1DGenerator(
#             n_colors, n_vertices_1d, self.backbone.out_channels,
#             color_share=lut1d_color_share)
#
#     def forward(self, img):
#         codes = self.backbone(img)
#         lut1d = self.lut1d_generator(codes)
#         iluts = []
#         for i in range(img.shape[0]):
#             iluts.append(torch.stack(
#                 torch.meshgrid(*(lut1d[i].unbind(0)[::-1]), indexing='ij'),
#                 dim=0).flip(0))
#         iluts = torch.stack(iluts, dim=0)
#         img = lut_transform(img, iluts)
#         lut3d_weights, lut3d = self.lut3d_generator(codes)
#         outs = lut_transform(img, lut3d)
#         return outs
#
#
# if __name__ == '__main__':
#     tensor = torch.randn(3, 3, 4, 4).cuda()
#     model = LUTNet('light').cuda()
#     out = model(tensor)
#     print(out.shape)
