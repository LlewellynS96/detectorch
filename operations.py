import torch
from itertools import repeat
from torch._six import container_abcs
import torch.nn as nn
import torch.nn.functional as F


EPSILON = 1e-8


class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return input.unsqueeze(self.dim)


def abs_c(x, keepdims=False):
    assert x.shape[1] == 2
    a = x[:, 0]
    b = x[:, 1]
    mag = torch.sqrt(torch.pow(a, 2) + torch.pow(b, 2) + EPSILON)
    if keepdims:
        mag = mag[:, None]
    return mag


def pair(x):
    if isinstance(x, container_abcs.Iterable):
        return x
    return tuple(repeat(x, 2))
