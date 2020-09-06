import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from operations import abs_c, pair


class ModReLu(nn.Module):
    def __init__(self, num_features, b=0, device='cuda'):
        super().__init__()
        self.b = Parameter(torch.Tensor(num_features).fill_(b))
        # self.device = device
        # self.b = torch.Tensor(num_features).fill_(b).to(self.device)

    def forward(self, input):
        mag = abs_c(input)
        arg = mag + self.b.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        mask = arg > 0.
        re = input[:, 0]
        im = input[:, 1]
        re = torch.where(mask, arg * re / mag, torch.zeros_like(re))
        im = torch.where(mask, arg * im / mag, torch.zeros_like(im))
        output = torch.cat((re.unsqueeze(1), im.unsqueeze(1)), 1)
        return output
