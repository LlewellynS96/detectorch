import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from operations import pair


class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        if self.affine:
            self.gamma = Parameter(torch.empty(2, 2, num_features))
            self.beta = Parameter(torch.empty(2, num_features))

        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

        self.register_buffer('running_mean', torch.empty(2, num_features))
        self.register_buffer('running_var', torch.empty(2, 2, num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

        self.reset_parameters()

    def reset_running_stats(self):
        with torch.no_grad():
            self.num_batches_tracked.zero_()
            self.running_mean.zero_()
            self.running_var.zero_()
            self.running_var[0, 0] = 1 / math.sqrt(2)
            self.running_var[1, 1] = 1 / math.sqrt(2)

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            with torch.no_grad():
                nn.init.zeros_(self.gamma)
                self.gamma[0, 0] = 1 / math.sqrt(2)
                self.gamma[1, 1] = 1 / math.sqrt(2)
                nn.init.zeros_(self.beta)

    def forward(self, input):

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / float(self.num_batches_tracked)
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        if self.training:
            mean = input.mean([0, -2, -1])

            input_center = input - mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            input_center_squared = torch.pow(input_center, 2)

            Vrr = input_center_squared[:, 0].mean([0, -2, -1]) + self.eps
            Vii = input_center_squared[:, 1].mean([0, -2, -1]) + self.eps
            Vri = (input_center[:, 0] * input_center[:, 1]).mean([0, -2, -1]) + self.eps
            V = torch.stack((Vrr, Vri, Vri, Vii)).reshape(2, 2, -1)
            with torch.no_grad():
                self.running_mean += exponential_average_factor * (mean - self.running_mean)
                self.running_var += exponential_average_factor * (V - self.running_var)
        else:
            mean = self.running_mean
            V = self.running_var

            input_center = input - mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        with torch.no_grad():
            tau = V[0, 0] + V[1, 1]
            delta = (V[0, 0] * V[1, 1]) - torch.pow(V[0, 1], 2)

            s = torch.sqrt(delta)
            t = torch.sqrt(tau + 2 * s)

            inverse_st = 1. / (s * t)
            Wrr = (V[1, 1] + s) * inverse_st
            Wii = (V[0, 0] + s) * inverse_st
            Wri = -V[0, 1] * inverse_st
            W = torch.stack((Wrr, Wri, Wri, Wii)).reshape(2, 2, -1)

        re = W[0, 0].unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * input_center[:, 0] + \
             W[0, 1].unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * input_center[:, 1]
        im = W[1, 0].unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * input_center[:, 0] + \
             W[1, 1].unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * input_center[:, 1]
        input_white = torch.cat((re.unsqueeze(1), im.unsqueeze(1)), 1)

        if self.affine:
            re = input_white[:, 0] * self.gamma[0, 0].unsqueeze(0).unsqueeze(-1).unsqueeze(-1) + \
                 input_white[:, 1] * self.gamma[0, 1].unsqueeze(0).unsqueeze(-1).unsqueeze(-1) + \
                 self.beta[0].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            im = input_white[:, 0] * self.gamma[1, 0].unsqueeze(0).unsqueeze(-1).unsqueeze(-1) + \
                 input_white[:, 1] * self.gamma[1, 1].unsqueeze(0).unsqueeze(-1).unsqueeze(-1) + \
                 self.beta[1].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            output = torch.cat((re.unsqueeze(1), im.unsqueeze(1)), 1)
        else:
            output = input_white

        return output

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**vars(self))

# class BatchNorm2d(nn.BatchNorm2d):
#     def forward(self, x):
#         self._check_input_dim(x)
#         y = x.transpose(0,1)
#         return_shape = y.shape
#         y = y.contiguous().view(x.size(1), -1)
#         mu = y.mean(dim=1)
#         sigma2 = y.var(dim=1)
#         if self.training is not True:
#             y = y - self.running_mean.view(-1, 1)
#             y = y / (self.running_var.view(-1, 1)**.5 + self.eps)
#         else:
#             if self.track_running_stats is True:
#                 with torch.no_grad():
#                     self.running_mean = (1-self.momentum)*self.running_mean + self.momentum*mu
#                     self.running_var = (1-self.momentum)*self.running_var + self.momentum*sigma2
#             y = y - mu.view(-1,1)
#             y = y / (sigma2.view(-1,1)**.5 + self.eps)
#
#         y = self.weight.view(-1, 1) * y + self.bias.view(-1, 1)
#         return y.view(return_shape).transpose(0,1)
