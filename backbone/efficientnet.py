import torch
import torch.nn as nn

class SE_Layer(nn.Module):
    def __init__(self, in_channel, expand_channel, reduction = 4):
        super(SE_Layer, self).__init__()
        squee_channel = int(in_channel/reduction)
        self.fc = nn.Sequential(
            nn.Conv2d(expand_channel, squee_channel, kernel_size=1, stride=1),
            nn.SiLU(),
            nn.Conv2d(squee_channel, expand_channel, kernel_size=1, stride=1),
            nn.SiLU()
        )

    def forward(self, x):
        scale = x.mean((2,3), keepdim = True)
        scale = self.fc(x)
        return scale * x

class Drop_Path(nn.Module):
    def __init__(self, drop_prob = None, training = False):
        super(Drop_Path, self).__init__()
        self.drop_prob = drop_prob
        self.training = training

    def forward(self, x):
        if self.drop_prob == 0. or self.training is False:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1.) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor()
        out = x.div(keep_prob) * random_tensor
        return out

class ConvBnAct(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size = 3,
                 stride = 1,
                 groups = 1,
                 norm_layer = None,
                 activati):
