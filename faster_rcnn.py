import torch
import pickle
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from detectron import FastRCNN, FasterRCNN, RPN
from dataset import PascalDatasetImage
from utils import to_numpy_image, add_bbox_to_image, index_dict_list, jaccard
from utils import NUM_WORKERS
import torchvision


def main():
    torch.random.manual_seed(12345)
    np.random.seed(12345)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    display = False
    display_rois = True
    train = True

    # faster_rcnn = FasterRCNN(anchors=[[45, 90], [64, 64], [90, 45],
    #                                   [90, 180], [128, 128], [180, 90],
    #                                   [180, 360], [256, 256], [360, 180],
    #                                   [360, 720], [512, 512], [720, 360]],
    #                          model_path='models/vgg11_bn-6002323d.pth',
    #                          device=device)
    faster_rcnn = FasterRCNN(anchors=[[233.1948, 570.4430],
                                      [103.9305, 419.0243],
                                      [402.4835, 463.7479],
                                      [109.3014,  86.1630],
                                      [116.5302, 208.1511],
                                      [ 35.6231,  59.8994],
                                      [ 55.4458, 164.8338],
                                      [670.7411, 365.8787],
                                      [316.4950, 320.4709],
                                      [181.8652, 330.0643],
                                      [652.4779, 551.9171],
                                      [238.8630, 162.8389],
                                      [713.4675, 740.7275],
                                      [541.8674, 221.0484],
                                      [418.1666, 689.3351]],
                             # model_path='models/vgg19_bn-c79401a0.pth',
                             model_path='models/vgg11_bn-6002323d.pth',
                             # model_path='models/vgg16-397923af.pth',
                             device=device)
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
                                  skip_difficult=False,
                                  image_size=(800, 800),
                                  do_transforms=False,
                                  )

    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=1,
                                  shuffle=True,
                                  num_workers=NUM_WORKERS)

    val_dataloader = DataLoader(dataset=val_data,
                                batch_size=1,
                                shuffle=False,
                                num_workers=NUM_WORKERS)

    # faster_rcnn = pickle.load(open('FasterR-CNN_working_RPN.pkl', 'rb'))
    faster_rcnn.to(device)

    if train:
        torch.random.manual_seed(12345)
        np.random.seed(12345)

        plist = [{'params': faster_rcnn.fast_rcnn.features.parameters(), 'lr': 1e-4},
                 {'params': faster_rcnn.rpn.parameters(), 'lr': 1e-4}
                 ]
        optimizer = optim.SGD(plist, momentum=0.9, weight_decay=0.0005)

        faster_rcnn.fit(train_data=train_data,
                        val_data=val_data,
                        optimizer=optimizer,
                        epochs=2,
                        batch_size=256,
                        multi_scale=False,
                        shuffle=True,
                        stage=0)

        plist = [{'params': faster_rcnn.fast_rcnn.parameters(), 'lr': 1e-4}]
        optimizer = optim.SGD(plist, momentum=0.9, weight_decay=0.0005)

        faster_rcnn.fit(train_data=train_data,
                        val_data=None,
                        optimizer=optimizer,
                        epochs=3,
                        batch_size=256,
                        multi_scale=False,
                        shuffle=True,
                        stage=1)

        plist = [{'params': faster_rcnn.rpn.parameters(), 'lr': 1e-4}]
        optimizer = optim.SGD(plist, momentum=0.9, weight_decay=0.0005)

        faster_rcnn.fit(train_data=train_data,
                        val_data=val_data,
                        optimizer=optimizer,
                        epochs=2,
                        batch_size=256,
                        multi_scale=False,
                        shuffle=True,
                        stage=2)

        plist = [{'params': faster_rcnn.fast_rcnn.classifier.parameters(), 'lr': 1e-4},
                 {'params': faster_rcnn.fast_rcnn.cls.parameters(), 'lr': 1e-4},
                 {'params': faster_rcnn.fast_rcnn.reg.parameters(), 'lr': 1e-4}]
        optimizer = optim.SGD(plist, momentum=0.9, weight_decay=0.0005)

        faster_rcnn.fit(train_data=train_data,
                        val_data=None,
                        optimizer=optimizer,
                        epochs=1,
                        batch_size=256,
                        multi_scale=False,
                        shuffle=True,
                        stage=1)

    if display_rois:
        torch.random.manual_seed(12345)
        np.random.seed(12345)

        with torch.no_grad():
            for images, image_info, image_transforms in train_dataloader:
                images = images.to(device)
                image_info_ = index_dict_list(image_info, 0)
                image_transforms_ = index_dict_list(image_transforms, 0)
                features = faster_rcnn.fast_rcnn.features(images)
                classes, bboxes = faster_rcnn.rpn(features)
                gt_bboxes, _ = train_data.get_gt_bboxes(image_info_, image_transforms_)
                gt_bboxes = gt_bboxes.to(device)
                ious = jaccard(bboxes, gt_bboxes)
                abs_max_iou, abs_argmax_iou = torch.max(ious, dim=0)
                classes = classes[abs_argmax_iou]
                bboxes = bboxes[abs_argmax_iou]
                width = image_info_['width']
                height = image_info_['height']
                image = to_numpy_image(images[0], size=(width, height))
                for cls, bbox, iou, i in zip(classes, bboxes, abs_max_iou, abs_argmax_iou):
                    name = i
                    add_bbox_to_image(image, bbox.squeeze(), iou, name)
                print('=' * 30)
                print('Median best bbox ranking: ', torch.median(abs_argmax_iou.float()).item())
                print('Median best bbox IoU: ', torch.median(abs_max_iou.float()).item())
                print('=' * 30)
                plt.imshow(image)
                plt.show()

    if display:
        with torch.no_grad():
            for images, image_info, image_transforms in train_dataloader:
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
