from models.blocks import *
from models.ttr import TTR
from models.translow import LFNet


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_features, in_features, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


def gauss_kernel(channels=3):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    return kernel


class LapPyramidConv(nn.Module):
    def __init__(self, num_high=4):
        super(LapPyramidConv, self).__init__()

        self.num_high = num_high
        self.kernel = gauss_kernel()

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def upsample(self, x):
        cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
        cc = cc.permute(0, 1, 3, 2)
        cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
        x_up = cc.permute(0, 1, 3, 2)
        return self.conv_gauss(x_up, 4 * self.kernel)

    def conv_gauss(self, img, kernel):
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        out = torch.nn.functional.conv2d(img, kernel.to(img.device), groups=img.shape[1])
        return out

    def pyramid_decom(self, img):
        current = img
        pyr = []
        for _ in range(self.num_high):
            filtered = self.conv_gauss(current, self.kernel)
            down = self.downsample(filtered)
            up = self.upsample(down)
            if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
                up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
            diff = current - up
            pyr.append(diff)
            current = down
        pyr.append(current)
        return pyr

    def pyramid_recons(self, pyr):
        image = pyr[-1]
        for level in reversed(pyr[:-1]):
            up = self.upsample(image)
            if up.shape[2] != level.shape[2] or up.shape[3] != level.shape[3]:
                up = nn.functional.interpolate(up, size=(level.shape[2], level.shape[3]))
            image = up + level
        return image


class TransHigh(nn.Module):
    def __init__(self, num_residual_blocks, num_high=3):
        super(TransHigh, self).__init__()

        self.num_high = num_high

        blocks = [nn.Conv2d(9, 64, 3, padding=1),
                  nn.LeakyReLU()]

        for _ in range(num_residual_blocks):
            blocks += [ResidualBlock(64)]

        blocks += [nn.Conv2d(64, 3, 3, padding=1)]

        self.model = nn.Sequential(*blocks)

        channels = 3
        # Stage1
        self.block1_1 = ConvLayer(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, dilation=2,
                                  norm=None, nonlinear='leakyrelu')
        self.block1_2 = ConvLayer(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, dilation=4,
                                  norm=None, nonlinear='leakyrelu')
        self.aggreation1_rgb = Aggreation(in_channels=channels * 3, out_channels=channels)
        # Stage2
        self.block2_1 = ConvLayer(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, dilation=8,
                                  norm=None, nonlinear='leakyrelu')
        self.block2_2 = ConvLayer(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, dilation=16,
                                  norm=None, nonlinear='leakyrelu')
        self.aggreation2_rgb = Aggreation(in_channels=channels * 3, out_channels=channels)
        # Stage3
        self.block3_1 = ConvLayer(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, dilation=32,
                                  norm=None, nonlinear='leakyrelu')
        self.block3_2 = ConvLayer(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, dilation=64,
                                  norm=None, nonlinear='leakyrelu')
        self.aggreation3_rgb = Aggreation(in_channels=channels * 3, out_channels=channels)
        # Stage3
        self.spp_img = SPP(in_channels=channels, out_channels=channels, num_layers=4, interpolation_type='bicubic')
        self.block4_1 = nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=1, stride=1)

    def forward(self, x, pyr_original, fake_low):
        pyr_result = [fake_low]
        mask = self.model(x)

        mask = nn.functional.interpolate(mask, size=(pyr_original[-2].shape[2], pyr_original[-2].shape[3]))
        result_highfreq = torch.mul(pyr_original[-2], mask) + pyr_original[-2]
        out1_1 = self.block1_1(result_highfreq)
        out1_2 = self.block1_2(out1_1)
        agg1_rgb = self.aggreation1_rgb(torch.cat((result_highfreq, out1_1, out1_2), dim=1))
        pyr_result.append(agg1_rgb)

        mask = nn.functional.interpolate(mask, size=(pyr_original[-3].shape[2], pyr_original[-3].shape[3]))
        result_highfreq = torch.mul(pyr_original[-3], mask) + pyr_original[-3]
        out2_1 = self.block2_1(result_highfreq)
        out2_2 = self.block2_2(out2_1)
        agg2_rgb = self.aggreation2_rgb(torch.cat((result_highfreq, out2_1, out2_2), dim=1))

        out3_1 = self.block3_1(agg2_rgb)
        out3_2 = self.block3_2(out3_1)
        agg3_rgb = self.aggreation3_rgb(torch.cat((agg2_rgb, out3_1, out3_2), dim=1))

        spp_rgb = self.spp_img(agg3_rgb)
        out_rgb = self.block4_1(spp_rgb)

        pyr_result.append(out_rgb)
        pyr_result.reverse()

        return pyr_result


class FilmNet(nn.Module):
    def __init__(self, depth=2):
        super(FilmNet, self).__init__()

        self.depth = depth
        self.lap_pyramid = LapPyramidConv(depth)

        self.trans_low = LFNet()
        self.lut = TTR()
        self.trans_high = TransHigh(3, num_high=depth)

    def forward(self, inp):
        pyr_inp = self.lap_pyramid.pyramid_decom(img=inp)
        out_low = self.trans_low(pyr_inp[-1])

        inp_up = nn.functional.interpolate(pyr_inp[-1], size=(pyr_inp[-2].shape[2], pyr_inp[-2].shape[3]))
        out_up = nn.functional.interpolate(out_low, size=(pyr_inp[-2].shape[2], pyr_inp[-2].shape[3]))
        high_with_low = torch.cat([pyr_inp[-2], inp_up, out_up], 1)

        pyr_inp_trans = self.trans_high(high_with_low, pyr_inp, out_low)

        result = self.lap_pyramid.pyramid_recons(pyr_inp_trans)

        result = self.lut(result)
        return result


if __name__ == '__main__':
    test = torch.randn(2, 3, 360, 540).cuda()
    model = FilmNet().cuda()
    out, loss = model(test, test)
    print(out.shape)
    print(loss)
