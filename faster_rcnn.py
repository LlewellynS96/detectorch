import torch
import torch.optim as optim
import numpy as np
import pickle
from detectron import FasterRCNN, ResNetBackbone
from miscellaneous import SSDataset
from torchvision import models
from utils import step_decay_scheduler, get_trainable_parameters, set_random_seed


def main():
    set_random_seed(12345)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    predict = True
    train = True
    joint = True

    ResNet = ResNetBackbone(model=models.resnet101,
                            model_path='models/resnet101-5d3b4d8f.pth',
                            pretrained=True,
                            input_dims=2,
                            device=device)

    faster_rcnn = FasterRCNN(name='FasterRCNN',
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
                             device=device).to(device)

    train_data = SSDataset(root_dir='../../../Data/SS/',
                           classes='../../../Data/SS/ss.names',
                           dataset='train',
                           mu=[0., 0.],
                           sigma=[0.348, 0.348],
                           mode='stft_iq',
                           return_targets=True
                           )

    val_data = SSDataset(root_dir='../../../Data/SS/',
                         classes='../../../Data/SS/ss.names',
                         dataset='val',
                         mu=[0., 0.],
                         sigma=[0.348, 0.348],
                         mode='stft_iq',
                         return_targets=True
                         )

    test_data = SSDataset(root_dir='../../../Data/SS/',
                          classes='../../../Data/SS/ss.names',
                          dataset='test',
                          mu=[0.141],
                          sigma=[0.469],
                          mode='stft',
                          return_targets=False
                          )

    if train:
        set_random_seed(12345)
        faster_rcnn.fast_rcnn.mini_freeze()
        plist = get_trainable_parameters(faster_rcnn)
        optimizer = optim.SGD(plist, lr=1e-3, momentum=0.9, weight_decay=5e-4, nesterov=True)
        scheduler = step_decay_scheduler(optimizer, steps=[-1, 200, 10000, 15000], scales=[.1, 10., 0.1, 0.1])
        losses = faster_rcnn.joint_training(train_data=train_data,
                                            val_data=val_data,
                                            optimizer=optimizer,
                                            scheduler=scheduler,
                                            epochs=18,
                                            shuffle=True,
                                            checkpoint_frequency=18
                                            )
        pickle.dump(losses, open('losses.pkl', 'wb'))

    if predict:
        set_random_seed(12345)
        faster_rcnn = pickle.load(open('FasterRCNN18.pkl', 'rb'))
        faster_rcnn.predict(dataset=test_data,
                            confidence_threshold=0.001,
                            overlap_threshold=0.5,
                            show=True,
                            export=False
                            )


if __name__ == '__main__':
    main()
