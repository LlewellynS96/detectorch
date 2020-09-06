import torch
import torch.optim as optim
import numpy as np
import pickle
import matplotlib.pyplot as plt
from detectron import FastRCNN, FasterRCNN, RPN, VGGBackbone, ResNetBackbone, ComplexResNetBackbone
from dataset import PascalDataset, SSDataset
from torchvision import models
from utils import step_decay_scheduler, to_numpy_image, add_bbox_to_image, get_trainable_parameters, jaccard, set_random_seed


def main():
    set_random_seed(12345)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    predict = True
    train = False
    joint = True

    #VGG = VGGBackbone(model=models.vgg16,
    #                  model_path='models/vgg16-397923af.pth',
    #                  pretrained=True,
    #                  device=device)

    ResNet = ResNetBackbone(model=models.resnet101,
                            model_path='models/resnet101-5d3b4d8f.pth',
                            pretrained=True,
                            input_dims=2,
                            device=device)

    # ResNet = ComplexResNetBackbone(device=device)

    #faster_rcnn = FasterRCNN(name='FasterRCNN',
    #                         #anchors=[[37,44], 
    #                         #         [61,132], 
    #                         #         [126,262], 
    #                         #         [136,82], 
    #                         #         [243,414], 
    #                         #         [279,169], 
    #                         #         [420,555], 
    #                         #         [581,266], 
    #                         #         [722,553]],
    #                         anchors=[ #  [45, 90], [64, 64], [90, 45],
    #                                  [90, 180], [128, 128], [180, 90],
    #                                  [180, 360], [256, 256], [360, 180],
    #                                  [360, 720], [512, 512], [720, 360]],
    #                         use_global_ctx=False,
    #                         backbone=VGG,
    #                         num_classes=3,
    #                         device=device)

    faster_rcnn = FasterRCNN(name='FasterRCNN',
                             #                         anchors=[[5, 29],
                             #                                  [22, 31],
                             #                                  [61, 41],
                             #                                  [73, 12],
                             #                                  [122, 42],
                             #                                  [169, 15],
                             #                                  [245, 43],
                             #                                  [589, 10],
                             #                                  [593, 47]],
                             anchors=[[121, 108],
                                      [147, 163],
                                      [176, 93],
                                      [185, 184],
                                      [254, 148],
                                      [261, 101],
                                      [335, 185],
                                      [342, 123],
                                      [466, 106]],
                             use_global_ctx=False,
                             backbone=ResNet,
                             num_classes=3,
                             device=device)

    faster_rcnn.to(device)

    train_data = SSDataset(root_dir='../../../Data/AMC/',
                           classes='../../../Data/AMC/amc.names',
                           dataset='train',
                           skip_truncated=False,
                           skip_difficult=False,
                           mu=[0., 0.],
                           sigma=[0.348, 0.348],
                           mode='stft_iq',
                           return_targets=True
                           )

    # test_data = SSDataset( root_dir='../../../Data/SS/',
    #                        classes='../../../Data/SS/ss.names',
    #                        dataset='test',
    #                        skip_truncated=False,
    #                        skip_difficult=False,
    #                        mu=[-25.52],
    #                        sigma=[8.55],
    #                        mode='spectrogram_ap_db',
    #                        return_targets=False
    #                        )

    # train_data = PascalDataset(#root_dir=['../Data/VOCdevkit/VOC2007/',
    #                                 #          '../Data/VOCdevkit/VOC2012/'],
    #                                 #classes='../Data/VOCdevkit/voc.names',
    #                                 #dataset=['trainval', 'trainval'],
    #                                 root_dir='../../../Data/AMC/',
    #                                 classes='../../../Data/AMC/amc.names',
    #                                 dataset='train',
    #                                 skip_difficult=False,
    #                                 skip_truncated=False,
    #                                 train=True,
    #                                 #mu=[0.485, 0.456, 0.406],
    #                                 mu=[0.174, 0.634, 0.505],
    #                                 #sigma=[0.229, 0.224, 0.225],
    #                                 sigma=[0.105, 0.068, 0.071],
    #                                 do_transforms=True,
    #                                 return_targets=True
    #                                 )
    #
    # val_data = PascalDataset(#root_dir='../Data/VOCdevkit/VOC2012/',
    #                               #classes='../Data/VOCdevkit/voc.names',
    #                               #dataset='val',
    #                               root_dir='../../../Data/SS/',
    #                               classes='../../../Data/SS/ss.names',
    #                               dataset='test',
    #                               skip_truncated=False,
    #                               train=True,
    #                               #mu=[0.485, 0.456, 0.406],
    #                               mu=[0.174, 0.634, 0.505],
    #                               #sigma=[0.229, 0.224, 0.225],
    #                               sigma=[0.105, 0.068, 0.071],
    #                               do_transforms=False,
    #                               return_targets=True
    #                               )
    #
    test_data = PascalDataset(#root_dir='../Data/VOCdevkit/VOC2007/',
                                   #classes='../Data/VOCdevkit/voc.names',
                                   #dataset='test',
                                   root_dir='../../../Data/SS/',
                                   classes='../../../Data/SS/ss.names',
                                   dataset='test',
                                   skip_difficult=False,
                                   skip_truncated=False,
                                   #mu=[0.485, 0.456, 0.406],
                                   mu=[0.174, 0.634, 0.505],
                                   #sigma=[0.229, 0.224, 0.225],
                                   sigma=[0.105, 0.068, 0.071],
                                   do_transforms=False
                                   )

    # faster_rcnn = pickle.load(open('models/FasterRCNNbest.pkl', 'rb'))

    if train:
        set_random_seed(12345)

        if joint:
            # faster_rcnn.fast_rcnn.mini_freeze()
            plist = get_trainable_parameters(faster_rcnn)
            optimizer = optim.SGD(plist, lr=1e-3, momentum=0.9, weight_decay=5e-4, nesterov=True)
            # optimizer = optim.Adam(plist, lr=1e-3, weight_decay=5e-4)
            # scheduler = step_decay_scheduler(optimizer, steps=[-1, 800, 96000, 128000], scales=[.1, 10., 0.1, 0.1])
            scheduler = step_decay_scheduler(optimizer, steps=[-1, 200, 10000, 15000], scales=[.1, 10., 0.1, 0.1])
            losses = faster_rcnn.joint_training(train_data=train_data,
                                                val_data=None,
                                                optimizer=optimizer,
                                                scheduler=scheduler,
                                                epochs=18,
                                                shuffle=True,
                                                checkpoint_frequency=18
                                                )
            pickle.dump(losses, open('losses.pkl', 'wb'))

        else:
            epochs = [15, 30, 10, 10]
            image_batch_size = [1, 4, 1, 12]
            roi_batch_size = [128, 128, 128, 128]

            faster_rcnn.alternate_training(train_data=train_data,
                                           val_data=train_data,
                                           epochs=epochs,
                                           lr=1e-1,
                                           image_batch_size=image_batch_size,
                                           roi_batch_size=roi_batch_size,
                                           shuffle=True,
                                           stage=None)

    if predict:
        torch.random.manual_seed(12345)
        np.random.seed(12345)

        faster_rcnn = pickle.load(open('FasterRCNN18.pkl', 'rb'))
        # faster_rcnn.fast_rcnn.backbone.channels = 512

        # faster_rcnn.rpn.predict(backbone=faster_rcnn.fast_rcnn.backbone,
        #                         dataset=test_data,
        #                         max_rois=10,
        #                         overlap_threshold=0.7,
        #                         show=True,
        #                         export=False)

        faster_rcnn.predict(dataset=test_data,
                            confidence_threshold=0.001,
                            overlap_threshold=.3,
                            show=False,
                            export=True
                            )


if __name__ == '__main__':
    main()
