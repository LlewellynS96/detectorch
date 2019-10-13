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

    faster_rcnn = FasterRCNN(name='FasterRCNN_Joint_Training',
                             anchors=[[45, 90], [64, 64], [90, 45],
                                      [90, 180], [128, 128], [180, 90],
                                      [180, 360], [256, 256], [360, 180],
                                      [360, 720], [512, 512], [720, 360]],
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

    if train:
        # torch.random.manual_seed(12345)
        # np.random.seed(12345)
        if joint:
            plist = [{'params': faster_rcnn.rpn.parameters()},
                     {'params': faster_rcnn.fast_rcnn.parameters()}]
            optimizer = optim.SGD(plist, lr=1e-4, momentum=0.9, weight_decay=0.0005)

            faster_rcnn.joint_training(train_data=train_data,
                                       val_data=val_data,
                                       optimizer=optimizer,
                                       max_rois=2000,
                                       image_batch_size=1,
                                       roi_batch_size=128,
                                       epochs=1,
                                       shuffle=True,
                                       multi_scale=True
                                       )

        else:
            epochs = [15, 30, 10, 10]
            image_batch_size = [1, 4, 1, 12]
            roi_batch_size = [128, 128, 128, 128]

            faster_rcnn.alternate_training(train_data=train_data,
                                           val_data=val_data,
                                           epochs=epochs,
                                           image_batch_size=image_batch_size,
                                           roi_batch_size=roi_batch_size,
                                           multi_scale=True,
                                           shuffle=True,
                                           stage=None)

    if predict:
        torch.random.manual_seed(12345)
        np.random.seed(12345)

        faster_rcnn = pickle.load(open('models/FasterRCNN_Alternate_Training_stage_3.pkl', 'rb'))

        faster_rcnn.predict(dataset=test_data,
                            batch_size=1,
                            confidence_threshold=0.6,
                            overlap_threshold=0.3,
                            show=True,
                            export=False
                            )


if __name__ == '__main__':
    main()
