import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log
from lib.Res2Net_v1b import res2net50_v1b_26w_4s


def cus_sample(feat, **kwargs):

    assert len(kwargs.keys()) == 1 and list(kwargs.keys())[0] in ["size", "scale_factor"]
    return F.interpolate(feat, **kwargs, mode="bilinear", align_corners=False)


class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=False, bn=True):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 最大池化

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 平均池化
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 最大池化
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ETFM(nn.Module):
    def __init__(self, channel=64):
        super(ETFM, self).__init__()
        self.conv1_2 = Conv1x1(channel*2, channel)
        self.conv3_3 = BasicConv2d(channel, channel, 3, padding=1)
        self.CA = ChannelAttention(channel//2)
        self.SA = SpatialAttention()
        self.upsample = cus_sample
        self.reduce = Conv1x1(32, 64)

    def forward(self, e, t, f):
        if e.size() != f.size():
            f = F.interpolate(f, e.size()[2:], mode='bilinear', align_corners=False)
        t = self.upsample(t, scale_factor=2)
        t = self.reduce(t)
        ef = e * f
        ef_1 = self.SA(ef)
        ef = ef_1 * ef
        tf = t * f
        tf_1 = self.SA(tf)
        tf = tf_1 * tf

        xe = torch.chunk(ef, 2, dim=1)
        xt = torch.chunk(tf, 2, dim=1)
        xf = torch.chunk(f, 2, dim=1)

        xe_0 = xe[0] + xf[0]
        xe_0_wei = self.CA(xe_0)
        xe_0 = xe_0 * xe_0_wei
        xe_1 = xe[1] + xf[1]
        xe_1_wei = self.CA(xe_1)
        xe_1 = xe_1 * xe_1_wei

        xt_0 = xt[0] + xf[0]
        xt_0_wei = self.SA(xt_0)
        xt_0 = xt_0 * xt_0_wei
        xt_1 = xt[1] + xf[1]
        xt_1_wei = self.CA(xt_1)
        xt_1 = xt_1 * xt_1_wei

        xf_0_0 = xe_0 + xe_1
        xf_0_1 = xe_0 * xe_1
        xf_0 = torch.cat((xf_0_0, xf_0_1), dim=1)
        xf_1_0 = xt_0 + xt_1
        xf_1_1 = xt_0 * xt_1
        xf_1 = torch.cat((xf_1_0, xf_1_1), dim=1)

        f_0_0 = xf_0 + xf_1
        f_1_1 = xf_0 * xf_1

        xx = self.conv1_2(torch.cat((f_0_0, f_1_1), dim=1))
        x = self.conv3_3(f + xx)

        return x


class EdgeEncoder(nn.Module):
    def __init__(self, channel):
        super(EdgeEncoder, self).__init__()
        self.CA = ChannelAttention(channel)
        self.upsample = cus_sample
        self.reduce1 = Conv1x1(256, 64)
        self.reduce4 = Conv1x1(2048, 64)
        self.conv3_3 = BasicConv2d(channel, channel, 3, padding=1)
        self.edge_cat = BasicConv2d(64+64, 64, kernel_size=3, padding=1)
        self.block = nn.Sequential(
            BasicConv2d(64, 64, 3, padding=1),
            BasicConv2d(64, 64, 3, padding=1),
            nn.Conv2d(64, 1, 3, padding=1))

    def forward(self, x, y):
        y = self.upsample(y, scale_factor=8)
        x = self.reduce1(x)
        y = self.reduce4(y)
        xy = self.conv3_3(self.edge_cat(torch.cat((x, y), dim=1)))
        wei = self.CA(xy)
        xy = xy * wei + x
        xy = self.block(xy)

        return xy


class CFM(nn.Module):
    def __init__(self, channel=64):
        super(CFM, self).__init__()
        self.conv1_2 = Conv1x1(channel, channel)
        self.conv3_3 = BasicConv2d(channel, channel, 3, padding=1)
        self.CA_1 = ChannelAttention(channel//2)
        self.CA = ChannelAttention(channel)

    def forward(self, lf, hf):
        if lf.size()[2:] != hf.size()[2:]:
            hf = F.interpolate(hf, size=lf.size()[2:], mode='bilinear', align_corners=False)
        xl = torch.chunk(lf, 2, dim=1)
        xh = torch.chunk(hf, 2, dim=1)
        x0 = xl[0] + xh[0]
        wei_0 = self.CA_1(x0)
        xl_0 = xl[0] * wei_0 + xh[0] * (1-wei_0)
        x1 = xl[1] + xh[1]
        wei_1 = self.CA_1(x1)
        xl_1 = xl[1] * wei_1 + xh[1] * (1-wei_1)
        xx = self.conv1_2(torch.cat((xl_0, xl_1), dim=1))
        xx_wei = self.CA(xx)
        xx = xx_wei * xx + lf
        xx = self.conv3_3(xx)

        return xx


class TextureEncoder(nn.Module):
    def __init__(self):
        super(TextureEncoder, self).__init__()
        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = BasicConv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = BasicConv2d(64, 32, kernel_size=3, stride=2, padding=1)
        self.conv_out = BasicConv2d(32, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.conv2(feat)
        xg = self.conv3(feat)
        pg = self.conv_out(xg)
        return xg, pg


class Network(nn.Module):
    def __init__(self, imagenet_pretrained=True):
        super(Network, self).__init__()
        self.resnet = res2net50_v1b_26w_4s(pretrained=imagenet_pretrained)
        # if self.training:
        # self.initialize_weights()
        self.rfb1_1 = RFB_modified(256, 64)
        self.rfb2_1 = RFB_modified(512, 64)
        self.rfb3_1 = RFB_modified(1024, 64)
        self.rfb4_1 = RFB_modified(2048, 64)

        self.upsample = cus_sample
        self.edge = EdgeEncoder(64)

        self.etfm1 = ETFM()
        self.etfm2 = ETFM()
        self.etfm3 = ETFM()
        self.etfm4 = ETFM()

        self.texture_encoder = TextureEncoder()
        self.cam1 = CFM()
        self.cam2 = CFM()
        self.cam3 = CFM()

        self.reduce2 = Conv1x1(64, 128)
        self.reduce3 = Conv1x1(64, 256)

        self.predictor1 = nn.Conv2d(64, 1, 3, padding=1)
        self.predictor2 = nn.Conv2d(128, 1, 3, padding=1)
        self.predictor3 = nn.Conv2d(256, 1, 3, padding=1)

    def forward(self, x):
        #x1, x2, x3, x4 = self.resnet(x)
        x0 = self.resnet.conv1(x)
        x0 = self.resnet.bn1(x0)
        x0 = self.resnet.relu(x0)
        x0 = self.resnet.maxpool(x0)
        x1 = self.resnet.layer1(x0)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        xg, pg = self.texture_encoder(x)

        x1_rfb = self.rfb1_1(x1)  # channel -> 64
        x2_rfb = self.rfb2_1(x2)  # channel -> 64
        x3_rfb = self.rfb3_1(x3)  # channel -> 64
        x4_rfb = self.rfb4_1(x4)  # channel -> 64

        edge = self.edge(x1, x4)
        edge_att = torch.sigmoid(edge)

        x1a = self.etfm1(edge_att, xg, x1_rfb)
        x2a = self.etfm2(edge_att, xg, x2_rfb)
        x3a = self.etfm3(edge_att, xg, x3_rfb)
        x4a = self.etfm4(edge_att, xg, x4_rfb)

        x34 = self.cam1(x3a, x4a)
        x234 = self.cam2(x2a, x34)
        x1234 = self.cam3(x1a, x234)
        x34 = self.reduce3(x34)
        x234 = self.reduce2(x234)

        o3 = self.predictor3(x34)
        o3 = F.interpolate(o3, scale_factor=4, mode='bilinear', align_corners=False)
        o2 = self.predictor2(x234)
        o2 = F.interpolate(o2, scale_factor=4, mode='bilinear', align_corners=False)
        o1 = self.predictor1(x1234)
        o1 = F.interpolate(o1, scale_factor=4, mode='bilinear', align_corners=False)
        oe = F.interpolate(edge_att, scale_factor=4, mode='bilinear', align_corners=False)
        ot = F.interpolate(pg, scale_factor=8, mode='bilinear', align_corners=False)

        return o3, o2, o1, oe, ot


if __name__ == '__main__':
    import numpy as np
    from time import time
    net = Network(imagenet_pretrained=False)
    net.eval()

    dump_x = torch.randn(1, 3, 352, 352)
    frame_rate = np.zeros((1000, 1))
    for i in range(1000):
        start = time()
        y = net(dump_x)
        end = time()
        running_frame_rate = 1 * float(1 / (end - start))
        print(i, '->', running_frame_rate)
        frame_rate[i] = running_frame_rate
    print(np.mean(frame_rate))
    print(y.shape)