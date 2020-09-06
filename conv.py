import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from operations import pair


class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = pair(kernel_size)
        self.stride = pair(stride)
        self.padding = pair(padding)
        self.dilation = pair(dilation)
        self.A_weight = Parameter(torch.Tensor(self.out_channels, self.in_channels, *self.kernel_size))
        self.B_weight = Parameter(torch.Tensor(self.out_channels, self.in_channels, *self.kernel_size))
        if bias:
            self.A_bias = Parameter(torch.Tensor(self.out_channels))
            self.B_bias = Parameter(torch.Tensor(self.out_channels))
        else:
            self.register_parameter('A_bias', None)
            self.register_parameter('B_bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.A_weight, a=math.sqrt(2))
        nn.init.kaiming_uniform_(self.B_weight, a=math.sqrt(2))
        if self.A_bias is not None:
            nn.init.zeros_(self.A_bias)
        if self.B_bias is not None:
            nn.init.zeros_(self.B_bias)

    def forward(self, input):
        re = F.conv2d(input[:, 0], self.A_weight, self.A_bias, self.stride, self.padding, self.dilation)
        re = re - F.conv2d(input[:, 1], self.B_weight, self.B_bias, self.stride, self.padding, self.dilation)
        im = F.conv2d(input[:, 0], self.B_weight, self.B_bias, self.stride, self.padding, self.dilation)
        im = im + F.conv2d(input[:, 1], self.A_weight, self.A_bias, self.stride, self.padding, self.dilation)
        output = torch.cat((re.unsqueeze(1), im.unsqueeze(1)), 1)
        return output


class ComplexConv2dSame(ComplexConv2d):
    """

    Notes
    -----
    This implementation was adapted from [1].

    [1] https://github.com/mlperf/inference/blob/master/others/edge/object_detection/ssd_mobilenet/pytorch/utils.py#L40
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True):

        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=0,
                         dilation=dilation,
                         bias=bias)

    def _compute_padding(self, input, dim):
        input_size = input.size(dim + 3)
        filter_size = self.A_weight.size(dim + 2)
        effective_filter_size = (filter_size - 1) * self.dilation[dim] + 1
        out_size = (input_size + self.stride[dim] - 1) // self.stride[dim]
        total_padding = max(
            0, (out_size - 1) * self.stride[dim] + effective_filter_size - input_size
        )
        additional_padding = int(total_padding % 2 != 0)

        return additional_padding, total_padding

    def forward(self, input):
        rows_odd, padding_rows = self._compute_padding(input, dim=0)
        cols_odd, padding_cols = self._compute_padding(input, dim=1)
        if rows_odd or cols_odd:
            input = F.pad(input, [0, cols_odd, 0, rows_odd])
        self.padding = (padding_rows // 2, padding_cols // 2)
        return super().forward(input)


class ComplexToRealConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = pair(kernel_size)
        self.stride = pair(stride)
        self.padding = pair(padding)
        self.dilation = pair(dilation)
        self.A_weight = Parameter(torch.Tensor(self.out_channels, self.in_channels, *self.kernel_size))
        self.B_weight = Parameter(torch.Tensor(self.out_channels, self.in_channels, *self.kernel_size))
        if bias:
            self.A_bias = Parameter(torch.Tensor(self.out_channels))
            self.B_bias = Parameter(torch.Tensor(self.out_channels))
        else:
            self.register_parameter('A_bias', None)
            self.register_parameter('B_bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.A_weight, a=math.sqrt(2))
        nn.init.kaiming_uniform_(self.B_weight, a=math.sqrt(2))
        if self.A_bias is not None:
            nn.init.zeros_(self.A_bias)
        if self.B_bias is not None:
            nn.init.zeros_(self.B_bias)

    def forward(self, input):
        re = F.conv2d(input[:, 0], self.A_weight, self.A_bias, self.stride, self.padding, self.dilation)
        im = F.conv2d(input[:, 1], self.B_weight, self.B_bias, self.stride, self.padding, self.dilation)
        output = re + im
        return output
