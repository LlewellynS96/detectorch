import torch
import torchvision
import time
import os
import copy
import torchsummary
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduction=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.logits:
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        f_loss = self.alpha * (1 - pt)**self.gamma * bce_loss

        if self.reduction is None or self.reduction == 'none':
            return f_loss
        elif self.reduction == 'mean':
            return torch.mean(f_loss)
        elif self.reduction == 'sum':
            return torch.sum(f_loss)
        else:
            raise AssertionError
