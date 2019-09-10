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
from backbone import FeatureExtractor


def main():
    torch.random.manual_seed(12345)
    np.random.seed(12345)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = FeatureExtractor(pretrained=True)
    model.to(device)

    torchsummary.summary(model, (3, 224, 224))


if __name__ == '__main__':
    main()