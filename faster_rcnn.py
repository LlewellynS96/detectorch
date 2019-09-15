import torch
import pickle
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from detectron import FastRCNN, FasterRCNN, RPN
from dataset import PascalDatasetImage
from utils import to_numpy_image, add_bbox_to_image, index_dict_list
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
                             device=device)
    faster_rcnn.to(device)

    torchvision.models.detection.fasterrcnn_resnet50_fpn

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

    faster_rcnn = pickle.load(open('FasterR-CNN_debug.pkl', 'rb'))

    if train:
        torch.random.manual_seed(12345)
        np.random.seed(12345)

        plist = [{'params': faster_rcnn.fast_rcnn.features.parameters(), 'lr': 1e-6},
                 {'params': faster_rcnn.rpn.parameters(), 'lr': 1e-4}
                 ]
        optimizer = optim.SGD(plist, lr=1e-4, momentum=0.9, weight_decay=5e-4)

        faster_rcnn.fit(train_data=train_data,
                        val_data=val_data,
                        optimizer=optimizer,
                        epochs=1,
                        multi_scale=False,
                        shuffle=True)

    if display_rois:
        torch.random.manual_seed(12345)
        np.random.seed(12345)

        with torch.no_grad():
            for images, image_info, _ in train_dataloader:
                images = images.to(device)
                image_info_ = index_dict_list(image_info, 0)
                features = faster_rcnn.fast_rcnn.features(images)
                classes, bboxes = faster_rcnn.rpn(features)
                width = image_info_['width']
                height = image_info_['height']
                image = to_numpy_image(images[0], size=(width, height))
                for cls, bbox in zip(classes, bboxes):
                    name = 'object'
                    add_bbox_to_image(image, bbox.squeeze(), cls[0], name)
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
