"""3D CNN encoder, Lite config (spec 7.2): stem 16, four stride-2 stages with two residual blocks each, GroupNorm with 8 groups, GELU."""
import torch.nn as nn


def conv_gn(ci, co, stride=1):
    return nn.Sequential(nn.Conv3d(ci, co, 3, stride, 1), nn.GroupNorm(8, co), nn.GELU())


class Res(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.a = conv_gn(c, c)
        self.b = nn.Sequential(nn.Conv3d(c, c, 3, 1, 1), nn.GroupNorm(8, c))
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.b(self.a(x)))


class Encoder(nn.Module):
    def __init__(self, chs=(16, 32, 64, 128, 128), in_ch=1):
        super().__init__()
        self.stem = conv_gn(in_ch, chs[0])
        self.stages = nn.ModuleList([nn.Sequential(conv_gn(chs[i], chs[i + 1], 2), Res(chs[i + 1]), Res(chs[i + 1])) for i in range(len(chs) - 1)])
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x, feats = self.stem(x), []
        for st in self.stages:
            x = st(x)
            feats.append(x)
        return feats  # S1..S4
