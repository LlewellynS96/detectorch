import torch
import torch.optim as optim
import numpy as np
import pickle
import matplotlib.pyplot as plt
from detectron import FastRCNN, FasterRCNN, RPN
from dataset import PascalDatasetImage
from utils import to_numpy_image, add_bbox_to_image, index_dict_list, jaccard


def main():
    torch.random.manual_seed(12345)
    np.random.seed(12345)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    predict = True
    train = True
    joint = True

    # faster_rcnn = FasterRCNN(name='FasterRCNN',
    #                          anchors=[[45, 90], [64, 64], [90, 45],
    #                                   [90, 180], [128, 128], [180, 90],
    #                                   [180, 360], [256, 256], [360, 180],
    #                                   [360, 720], [512, 512], [720, 360]],
    #                          model_path='models/vgg16-397923af.pth',
    #                          device=device)
    faster_rcnn = FasterRCNN(name='FasterRCNN',
                             anchors=[[123.4950, 100.2833],
                                      [212.2724, 189.5562],
                                      [365.2326, 410.2795],
                                      [215.3095, 351.9713],
                                      [232.9199, 590.1595],
                                      [435.3557, 659.2862],
                                      [64.1715, 175.1229],
                                      [38.7346, 63.9380],
                                      [659.8110, 446.5352],
                                      [520.1739, 238.6301],
                                      [112.8695, 340.1479],
                                      [710.3906, 715.7178]],
                             model_path='models/vgg16-397923af.pth',
                             device=device)
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

    train_data = PascalDatasetImage(root_dir='../data/VOC2012/',
                                    classes='../data/VOC2012/voc.names',
                                    dataset='train',
                                    skip_truncated=False,
                                    skip_difficult=False,
                                    image_size=(800, 800),
                                    do_transforms=True,
                                    )

    val_data = PascalDatasetImage(root_dir='../data/VOC2012/',
                                  classes='../data/VOC2012/voc.names',
                                  dataset='val',
                                  skip_truncated=False,
                                  skip_difficult=True,
                                  image_size=(800, 800),
                                  do_transforms=False,
                                  )

    test_data = PascalDatasetImage(root_dir='../data/VOC2007/',
                                   classes='../data/VOC2012/voc.names',
                                   dataset='test',
                                   skip_truncated=False,
                                   skip_difficult=False,
                                   image_size=(800, 800),
                                   do_transforms=False
                                   )

    faster_rcnn = pickle.load(open('FasterRCNN6.pkl', 'rb'))

    if train:
        torch.random.manual_seed(12345)
        np.random.seed(12345)
        if joint:
            plist = [{'params': faster_rcnn.parameters()}]
            optimizer = optim.SGD(plist, lr=1e-2, momentum=0.9, weight_decay=1e-4, nesterov=True)

            target_lr = optimizer.defaults['lr']
            initial_lr = 1e-3
            warm_up = 2
            step_size = 0.95
            step_frequency = 1
            gradient = (target_lr - initial_lr) / warm_up

            def f(e):
                if e < warm_up:
                    return gradient * e + initial_lr
                else:
                    return target_lr * step_size ** ((e - warm_up) // step_frequency)

            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)

            faster_rcnn.joint_training(train_data=train_data,
                                       val_data=None,
                                       optimizer=optimizer,
                                       scheduler=scheduler,
                                       image_batch_size=1,
                                       roi_batch_size=128,
                                       epochs=2,
                                       shuffle=True,
                                       multi_scale=False,
                                       checkpoint_frequency=1
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

        faster_rcnn.predict(dataset=test_data,
                            batch_size=1,
                            confidence_threshold=0.4,
                            overlap_threshold=.45,
                            show=True,
                            export=False
                            )


if __name__ == '__main__':
    main()
