import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from operations import abs_c


class ComplexMaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode

    def forward(self, input):
        mag = abs_c(input)
        _, idx = F.max_pool2d(mag, self.kernel_size, self.stride,
                              self.padding, self.dilation, self.ceil_mode,
                              True)
        re = input[:, 0].flatten()[idx]
        im = input[:, 1].flatten()[idx]
        output = torch.cat((re.unsqueeze(1), im.unsqueeze(1)), 1)
        return output


class ComplexAdaptiveAvgPool2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input.mean([-2, -1]).unsqueeze(1).unsqueeze(-1)
        return output
