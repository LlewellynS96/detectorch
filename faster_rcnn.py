import torch
import torch.optim as optim
import numpy as np
import pickle
import matplotlib.pyplot as plt
from detectron import FastRCNN, FasterRCNN, RPN, VGGBackbone
from dataset import PascalDatasetImage
from torchvision import models
from utils import step_decay_scheduler, to_numpy_image, add_bbox_to_image, access_dict_list, jaccard, set_random_seed


def main():
    set_random_seed(12345)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    predict = True
    train = True
    joint = True

    VGG = VGGBackbone(model=models.vgg11_bn(),
                      model_path='models/vgg11_bn-6002323d.pth',
                      device=device)

    faster_rcnn = FasterRCNN(name='FasterRCNN',
                             anchors=[[45, 90], [64, 64], [90, 45],
                                      [90, 180], [128, 128], [180, 90],
                                      [180, 360], [256, 256], [360, 180],
                                      [360, 720], [512, 512], [720, 360]],
                             backbone=VGG,
                             device=device)

    # faster_rcnn = FasterRCNN(name='FasterRCNN',
    #                          anchors=[[38.7346, 63.9380],
    #                                   [64.1715, 175.1229],
    #                                   [112.8695, 340.1479],
    #                                   [123.4950, 100.2833],
    #                                   [212.2724, 189.5562],
    #                                   [215.3095, 351.9713],
    #                                   [232.9199, 590.1595],
    #                                   [365.2326, 410.2795],
    #                                   [435.3557, 659.2862],
    #                                   [520.1739, 238.6301],
    #                                   [659.8110, 446.5352],
    #                                   [710.3906, 715.7178]],
    #                          backbone=VGG,
    #                          device=device)

    # faster_rcnn = FasterRCNN(anchors=[[233.1948, 570.4430],
    #                                   [103.9305, 419.0243],
    #                                   [402.4835, 463.7479],
    #                                   [109.3014,  86.1630],
    #                                   [116.5302, 208.1511],
    #                                   [ 35.6231,  59.8994],
    #                                   [ 55.4458, 164.8338],
    #                                   [670.7411, 365.8787],
    #                                   [316.4950, 320.4709],
    #                                   [181.8652, 330.0643],
    #                                   [652.4779, 551.9171],
    #                                   [238.8630, 162.8389],
    #                                   [713.4675, 740.7275],
    #                                   [541.8674, 221.0484],
    #                                   [418.1666, 689.3351]],
    #                          model_path='models/vgg16-397923af.pth',
    #                          device=device)
    faster_rcnn.to(device)

    train_data = PascalDatasetImage(root_dir=['../../../Data/VOCdevkit/VOC2012/'],
                                    # '../../../Data/VOCdevkit/VOC2012/'],
                                    classes='../../../Data/VOCdevkit/voc.names',
                                    # dataset=['trainval', 'trainval'],
                                    dataset=['train'],
                                    skip_difficult=False,
                                    skip_truncated=False,
                                    mu=[0.458, 0.438, 0.405],  # Maybe reverse the order?
                                    sigma=[0.247, 0.242, 0.248],  # A bit high?
                                    do_transforms=True,
                                    return_targets=True
                                    )

    val_data = PascalDatasetImage(root_dir='../../../Data/VOCdevkit/VOC2012/',
                                  classes='../../../Data/VOCdevkit/voc.names',
                                  dataset='val',
                                  skip_truncated=False,
                                  mu=[0.458, 0.438, 0.405],
                                  sigma=[0.247, 0.242, 0.248],
                                  do_transforms=False,
                                  )

    test_data = PascalDatasetImage(root_dir='../../../Data/VOCdevkit/VOC2007/',
                                   classes='../../../Data/VOCdevkit/voc.names',
                                   dataset='test',
                                   skip_truncated=False,
                                   mu=[0.458, 0.438, 0.405],
                                   sigma=[0.247, 0.242, 0.248],
                                   do_transforms=False
                                   )

    # faster_rcnn = pickle.load(open('models/FasterRCNNbest.pkl', 'rb'))

    if train:
        set_random_seed(12345)

        if joint:
            plist = [{'params': faster_rcnn.parameters()}]
            optimizer = optim.SGD(plist, lr=1e-3, momentum=0.9, weight_decay=1e-4, nesterov=True)
            # optimizer = optim.Adam(plist, lr=1e-3, weight_decay=1e-4)
            scheduler = step_decay_scheduler(optimizer, steps=[-1, 2000, 60000, 90000], scales=[.1, 10., 0.1, 0.1])

            faster_rcnn.joint_training(train_data=train_data,
                                       val_data=None,
                                       optimizer=optimizer,
                                       scheduler=scheduler,
                                       epochs=20,
                                       shuffle=True,
                                       checkpoint_frequency=5
                                       )

        else:
            epochs = [15, 30, 10, 10]
            image_batch_size = [1, 4, 1, 12]
            roi_batch_size = [128, 128, 128, 128]

            faster_rcnn.alternate_training(train_data=train_data,
                                           val_data=val_data,
                                           epochs=epochs,
                                           lr=1e-1,
                                           image_batch_size=image_batch_size,
                                           roi_batch_size=roi_batch_size,
                                           multi_scale=True,
                                           shuffle=True,
                                           stage=None)

    if predict:
        torch.random.manual_seed(12345)
        np.random.seed(12345)

        faster_rcnn = pickle.load(open('models/FasterRCNN20.pkl', 'rb'))

        faster_rcnn.predict(dataset=test_data,
                            confidence_threshold=0.001,
                            overlap_threshold=.3,
                            show=False,
                            export=True
                            )


if __name__ == '__main__':
    main()
