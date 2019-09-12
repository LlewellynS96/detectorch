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
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from detectron import FastRCNN, RPN
from dataset import PascalDatasetImage
from utils import to_numpy_image, add_bbox_to_image, index_dict_list


def main():
    torch.random.manual_seed(12345)
    np.random.seed(12345)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # backbone = FastRCNN(pretrained=True)
    # backbone.to(device)
    #
    # torchsummary.summary(backbone.features, (3, 224, 224))
    # torchsummary.summary(backbone.classifier, (1, 25088))

    dataset = PascalDatasetImage(root_dir='/home/llewellyn/developer/volume/data/VOC2007/',
                                 classes='../data/VOC2012/voc.names',
                                 dataset='test',
                                 skip_truncated=False,
                                 skip_difficult=False,
                                 image_size=(416, 416),
                                 do_transforms=True,
                                 )

    dataloader = DataLoader(dataset=dataset,
                            batch_size=10,
                            shuffle=False,
                            num_workers=0)

    with torch.no_grad():
        for images, image_info, image_transforms in dataloader:
            for idx, image in enumerate(images):
                image_info_ = index_dict_list(image_info, idx)
                image_transforms_ = index_dict_list(image_transforms, idx)
                bboxes, classes = dataset.get_gt_bboxes(image_info_, image_transforms_)
                width = image_info_['width']
                height = image_info_['height']
                image = to_numpy_image(image, size=(width, height))
                for bbox, cls in zip(bboxes, classes):
                    name = dataset.decode_categorical(cls)
                    add_bbox_to_image(image, bbox, 1., name)
                plt.imshow(image)
                plt.show()


if __name__ == '__main__':
    main()
