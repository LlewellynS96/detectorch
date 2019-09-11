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
from torchvision import datasets, models, transforms, ops
import matplotlib.pyplot as plt


class FastRCNN(nn.Module):

    def __init__(self, num_classes=20, pretrained=True, model_path='models/vgg11_bn-6002323d.pth'):

        super(FastRCNN, self).__init__()
        self.num_classes = num_classes
        self.model = models.vgg11_bn()
        if pretrained:
            self.model_path = model_path
            print('Loading pretrained weights from {}'.format(self.model_path))
            state_dict = torch.load(self.model_path)
            self.model.load_state_dict({k: v for k, v in state_dict.items() if k in self.model.state_dict()})
        self.model.features = nn.Sequential(*list(self.model.features.children())[:-1])
        self.model.classifier = nn.Sequential(*list(self.model.classifier.children())[:-1])

        self.features = self.model.features
        self.roi_pool = ops.RoIPool(output_size=7, spatial_scale=1.)
        self.classifier = self.model.classifier
        self.head = {'cls': nn.Linear(4096, self.num_classes + 1),
                     'reg': nn.Linear(4096, self.num_classes * 4)}

    def forward(self, x):

        pass

    def freeze(self):
        """
            Freezes the backbone (conv and bn) of the VGG network.
        """
        for param in self.features.parameters():
            param.requires_grad = False


class RPN(nn.Module):

    def __init__(self, num_anchors):

        super(RPN, self).__init__()
        self.in_channels = 512
        self.num_features = 256
        self.num_anchors = num_anchors
        self.windows = nn.Sequential(nn.Conv2d(in_channels=self.in_channels,
                                               out_channels=self.num_features,
                                               kernel_size=3,
                                               stride=1,
                                               padding=1),
                                     nn.ReLU()
                                     )

    def forward(self, x):

        features = self.windows(x)
        cls = nn.Conv2d(in_channels=self.num_features,
                        out_channels=2 * self.num_anchors,
                        kernel_size=1)(features)
        cls = nn.Softmax(cls, dim=1)
        reg = nn.Conv2d(in_channels=self.num_features,
                        out_channels=4 * self.num_anchors,
                        kernel_size=1)(features)

        return cls, reg
