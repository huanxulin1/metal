import torch
import torch.nn as nn

class Conv(nn.Module):
    def __init__(self, c1, c2, k = 1, s = 1, p = None, g = 1, act = True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups= g, bias= False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class Foucs(nn.Module):
    def __init__(self, c1, c2, k = 1, s = 1, p = None, g = 1, act = True):
        super(Foucs, self).__init__()
        self.conv = nn.Conv2d(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1:2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))

class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut = True, g = 1, e = 0.5):
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)
        self.conv1 = Conv(c1, c_, k = 1, s = 1)
        self.conv2 = Conv(c_, c2, k = 3, s = 1, g = g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))

class CSP3(nn.Module):
    def __init__(self, c1, c2, n = 1, shortcut = True, g = 1, e = 0.5):
        super(CSP3, self).__init__()
        c_ = int(c2 * e)
        self.conv1 = Conv(c1, c_, 1, 1)
        self.conv2 = Conv(c1, c_, 1, 1)
        self.conv3 = Conv(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e) for _ in range(n)])

    def forward(self, x):
        return self.conv3(torch.cat(self.m(self.conv1(x)), self.conv2(x)))
