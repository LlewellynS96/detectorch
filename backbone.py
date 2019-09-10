import torch
import torchvision
import time
import os
import copy
import torchsummary
import numpy as np
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt


class FeatureExtractor(nn.Module):

    def __init__(self, pretrained=True):

        super(FeatureExtractor, self).__init__()
        self.vgg = models.vgg11_bn(pretrained=pretrained)
        self.vgg.features = nn.Sequential(*list(self.vgg.features.children())[:-1])

    def forward(self, x):

        return self.vgg.features(x)


