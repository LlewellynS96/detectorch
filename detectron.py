import torch
import torchvision
import pickle
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
from torchvision import models, ops
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import jaccard, random_choice, parameterize_bboxes, deparameterize_bboxes, index_dict_list, export_prediction
from utils import to_numpy_image, add_bbox_to_image, nullcontext
from utils import NUM_WORKERS
from layers import FocalLoss


NETWORK_STRIDE = 16
RPN_HI_THRESHOLD = 0.7
RPN_LO_THRESHOLD = 0.3
FRCN_HI_THRESHOLD = 0.5
FRCN_LO_LO_THRESHOLD = 0.1
FRCN_LO_HI_THRESHOLD = 0.3
RPN_POS_RATIO = 0.5
FRCN_POS_RATIO = 0.375
RPN_NMS_THRESHOLD = 0.7
MAX_TRAIN_RPN_ROIS = 2000
MAX_TEST_RPN_ROIS = 300


class FastRCNN(nn.Module):

    def __init__(self, num_classes=20, pretrained=True, model_path='models/vgg11_bn-6002323d.pth', device='cuda'):

        super(FastRCNN, self).__init__()
        self.num_classes = num_classes
        self.device = device
        self.model = models.vgg11_bn()
        if pretrained:
            self.model_path = model_path
            print('Loading pretrained weights from {}'.format(self.model_path))
            state_dict = torch.load(self.model_path)
            self.model.load_state_dict({k: v for k, v in state_dict.items() if k in self.model.state_dict()})
        self.model.features = nn.Sequential(*list(self.model.features.children())[:-1])
        self.model.classifier = nn.Sequential(*list(self.model.classifier.children())[:-1])

        self.features = self.model.features
        self.classifier = self.model.classifier
        self.cls = nn.Sequential(nn.Linear(4096, self.num_classes + 1),
                                 nn.Softmax(dim=-1))
        self.reg = nn.Linear(4096, self.num_classes * 4)

        self.to(device)

    def forward(self, x, rois, extract_features=True):
        batch_size = x.shape[0]
        if extract_features:
            features = self.features(x)
        else:
            features = x
        grid_size = features.shape[-1]
        roi_features = ops.roi_pool(features, rois, output_size=(7, 7), spatial_scale=grid_size)
        classifier = self.classifier(roi_features.view(batch_size, -1, 512 * 7 * 7))
        cls = self.cls(classifier)
        reg = self.reg(classifier)

        reg = reg.reshape(batch_size, -1, self.num_classes, 4)

        return cls, reg

    def freeze(self):
        """
            Freezes the backbone (conv and bn) of the VGG network.
        """
        for param in self.features.parameters():
            param.requires_grad = False

    def training(self, train_data, optimizer, backbone=None, rpn=None, rois=None, max_rois=2000, image_batch_size=2,
                 roi_batch_size=32, epochs=1, val_data=None, shuffle=True, multi_scale=True, checkpoint_frequency=100):

        assert rpn or rois

        self.train()

        train_dataloader = DataLoader(dataset=train_data,
                                      batch_size=image_batch_size,
                                      shuffle=shuffle,
                                      num_workers=NUM_WORKERS)

        train_loss = []
        val_loss = []

        for epoch in range(1, epochs + 1):
            batch_loss = []
            if multi_scale:
                random_size = np.random.randint(49, 52) * NETWORK_STRIDE
                train_data.set_image_size(random_size, random_size)
            with tqdm(total=len(train_dataloader),
                      desc='Epoch: [{}/{}]'.format(epoch, epochs),
                      leave=True,
                      unit='batches') as inner:
                for images, images_info, images_transforms in train_dataloader:
                    images = images.to(self.device)
                    image_batch_size = len(images)
                    target_rois = [torch.zeros(roi_batch_size, 4, device=self.device)] * image_batch_size
                    target_classes = torch.zeros(image_batch_size, roi_batch_size, self.num_classes + 1, device=self.device)
                    target_bboxes = torch.zeros(image_batch_size, roi_batch_size, 4, device=self.device)
                    with torch.no_grad():
                        if backbone is None:
                            features = self.features(images)
                        else:
                            features = backbone(images)
                    for i in range(image_batch_size):
                        image_info_ = index_dict_list(images_info, i)
                        image_transforms_ = index_dict_list(images_transforms, i)
                        image_bboxes, image_classes = train_data.get_gt_bboxes(image_info_, image_transforms_)
                        image_bboxes = image_bboxes.to(self.device)
                        image_classes = image_classes.to(self.device)
                        if image_bboxes.numel() > 0:
                            if rpn is not None:
                                cls, reg = rpn(features[i][None])
                                grid_size = torch.tensor(features.shape[-2:], device=self.device)
                                rois = rpn.post_process(cls, reg, grid_size, max_rois, True)[1][0]
                            ious = jaccard(rois, image_bboxes)
                            max_iou, argmax_iou = torch.max(ious, dim=1)
                            positive_mask = max_iou > FRCN_HI_THRESHOLD
                            abs_max_iou = torch.argmax(ious, dim=0)
                            positive_mask[abs_max_iou] = 1  # Added ROI with maximum IoU to positive samples.
                            positive_mask = torch.nonzero(positive_mask)
                            negative_mask = torch.nonzero((FRCN_LO_LO_THRESHOLD < max_iou) * (max_iou < FRCN_LO_HI_THRESHOLD))
                            if negative_mask.numel() > 0:
                                samples = random_choice(negative_mask, roi_batch_size, replace=True)
                            else:
                                negative_mask = torch.nonzero(max_iou < FRCN_LO_HI_THRESHOLD)
                                if negative_mask.numel() > 0:
                                    samples = random_choice(negative_mask, roi_batch_size, replace=True)
                                else:
                                    negative_mask = torch.nonzero(max_iou < FRCN_HI_THRESHOLD)
                                    samples = random_choice(negative_mask, roi_batch_size, replace=True)
                            frcn_pos_numel = int(roi_batch_size * FRCN_POS_RATIO)
                            if positive_mask.numel() < frcn_pos_numel:
                                frcn_pos_numel = positive_mask.numel()
                            if frcn_pos_numel > 0:
                                samples[:frcn_pos_numel] = random_choice(positive_mask, frcn_pos_numel)
                            target_rois[i] = rois[samples]
                            target_classes[i][:frcn_pos_numel] = image_classes[argmax_iou[samples[:frcn_pos_numel]]]
                            target_classes[i][frcn_pos_numel:, 0] = 1.
                            target_bboxes[i] = parameterize_bboxes(image_bboxes[argmax_iou[samples]], rois[samples])
                    optimizer.zero_grad()
                    cls, reg = self(images, target_rois)
                    loss = self.loss(cls, reg, target_classes.detach(), target_bboxes.detach())
                    batch_loss.append(loss['total'].item())
                    loss['total'].backward()
                    optimizer.step()
                    inner.set_postfix_str(' Training Loss: {:.6f}'.format(np.mean(batch_loss)))
                    inner.update()
                train_loss.append(np.mean(batch_loss))
                if val_data is not None:
                    val_data.reset_image_size()
                    val_loss.append(self.calculate_loss(val_data, backbone, batch_size))
                    inner.set_postfix_str(' Training Loss: {:.6f},  Validation Loss: {:.6f}'.format(train_loss[-1],
                                                                                                    val_loss[-1]))
                else:
                    inner.set_postfix_str(' Training Loss: {:.6f}'.format(train_loss[-1]))

        return train_loss, val_loss

    @staticmethod
    def loss(cls, reg, target_cls, target_reg, focal=True):

        loss = dict()
        lambd = 1. / cls.shape[0] / cls.shape[1]

        if focal:
            loss['cls'] = lambd * FocalLoss(reduction='sum')(cls, target_cls)
        else:
            loss['cls'] = nn.BCELoss(reduction='sum')(cls, target_cls)

        obj_mask = torch.nonzero(target_cls[:, :, 0] == 0)
        arange = torch.arange(len(obj_mask))
        obj_mask = tuple(obj_mask.t())
        if len(obj_mask) > 0:
            cls = torch.argmax(target_cls, dim=-1) - 1.
            loss['reg'] = lambd * nn.SmoothL1Loss(reduction='sum')(reg[obj_mask][arange, cls[obj_mask]],
                                                                   target_reg[obj_mask])
        else:
            loss['reg'] = 0.

        loss['total'] = loss['cls'] + loss['reg']
        return loss

    def post_process(self, cls, reg, rois):
        batch_size = cls.shape[0]

        classes = []
        bboxes = []

        for i in range(batch_size):
            cls_argmax = torch.argmax(cls[i], dim=-1) - 1.
            mask = cls_argmax >= 0
            cls_argmax = cls_argmax[mask]
            cls_ = cls[i][mask]
            arange = torch.arange(len(cls_))
            reg_ = reg[i][mask][arange, cls_argmax]
            rois_ = rois[i][mask]

            if reg_.numel() > 0:
                reg_ = deparameterize_bboxes(reg_, rois_)
                reg_ = torch.clamp(reg_, min=0., max=1.)

                scores = cls_[:, 0]
                sort = torch.argsort(scores, descending=True)
                classes.append(cls_[sort])
                bboxes.append(reg_[sort])
            else:
                classes.append(torch.tensor([], device=self.device))
                bboxes.append(torch.tensor([], device=self.device))

        return classes, bboxes

    def process_bboxes(self, cls, reg, confidence_threshold=0.01, overlap_threshold=0.5, nms=True):

        image_idx_ = []
        bboxes_ = []
        confidence_ = []
        classes_ = []

        for i, (classes, bboxes) in enumerate(zip(cls, reg)):

            if classes.numel() == 0:
                continue

            confidence, classes = torch.max(classes, dim=-1)

            mask = confidence > confidence_threshold

            if sum(mask) == 0:
                continue

            bboxes = bboxes[mask]
            confidence = confidence[mask]
            classes = classes[mask]

            if nms:
                cls = torch.unique(classes)
                for c in cls:
                    cls_mask = (classes == c).nonzero().flatten()
                    mask = torchvision.ops.nms(bboxes[cls_mask], confidence[cls_mask], overlap_threshold)
                    bboxes_.append(bboxes[cls_mask][mask])
                    confidence_.append(confidence[cls_mask][mask])
                    classes_.append(classes[cls_mask][mask])
                    image_idx_.append(torch.ones(len(bboxes[cls_mask][mask]), device=self.device) * i)
            else:
                bboxes_.append(bboxes)
                confidence_.append(confidence)
                classes_.append(classes)
                image_idx_.append(torch.ones(len(bboxes)) * i)

        if len(bboxes_) > 0:
            bboxes = torch.cat(bboxes_).view(-1, 4)
            classes = torch.cat(classes_).flatten()
            confidence = torch.cat(confidence_).flatten()
            image_idx = torch.cat(image_idx_).flatten()

            return bboxes, classes, confidence, image_idx
        else:
            return torch.tensor([], device=self.device), \
                   torch.tensor([], device=self.device), \
                   torch.tensor([], device=self.device), \
                   torch.tensor([], device=self.device)


class RPN(nn.Module):

    def __init__(self, anchors, device='cuda'):

        super(RPN, self).__init__()
        self.in_channels = 512
        self.num_features = 512
        self.device = device
        self.anchors = torch.tensor(anchors, device=self.device, dtype=torch.float) / NETWORK_STRIDE
        self.num_anchors = len(anchors)
        self.sliding_windows = nn.Conv2d(in_channels=self.in_channels,
                                         out_channels=self.num_features,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1)
        self.cls = nn.Conv2d(in_channels=self.num_features,
                             out_channels=2 * self.num_anchors,
                             kernel_size=1)
        self.reg = nn.Conv2d(in_channels=self.num_features,
                             out_channels=4 * self.num_anchors,
                             kernel_size=1)

        self.to(device)

    def forward(self, x):
        batch_size = x.shape[0]
        features = F.relu(self.sliding_windows(x))
        cls = self.cls(features)
        reg = self.reg(features)

        cls = cls.permute(0, 2, 3, 1).reshape(batch_size, -1, 2)
        cls = torch.softmax(cls, dim=-1)
        reg = reg.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)

        return cls, reg

    def post_process(self, cls, reg, grid_size, max_rois=2000, nms=False):
        anchors = self.construct_anchors(grid_size)

        batch_size = cls.shape[0]
        confidence = torch.zeros(batch_size, max_rois, 2, device=self.device)
        bboxes = []

        for i in range(batch_size):
            reg_ = deparameterize_bboxes(reg[i], anchors)
            reg_ = torch.clamp(reg_, min=0., max=1.)

            scores = cls[i, :, 0]
            if nms:
                keep = torchvision.ops.nms(reg_, scores, RPN_NMS_THRESHOLD)
            else:
                keep = torch.argsort(scores, descending=True)
            confidence[i] = cls[i][keep][:max_rois]
            bboxes.append(reg_[keep][:max_rois])

        return confidence, bboxes

    def construct_anchors(self, grid_size):

        anchors = torch.zeros((*grid_size, self.num_anchors, 4), device=self.device)

        x = torch.arange(0, grid_size[0], device=self.device)
        y = torch.arange(0, grid_size[1], device=self.device)

        xx, yy = torch.meshgrid(x, y)

        x_coords = x.clone().reshape(-1, 1, 1, 1).float()
        y_coords = y.clone().reshape(1, -1, 1, 1).float()

        anchors[xx, yy, :, :2] = -self.anchors / 2.
        anchors[xx, yy, :, 2:] = self.anchors / 2.

        anchors[x, :, :, ::2] += x_coords
        anchors[x, :, :, ::2] /= grid_size[0].float()
        anchors[:, y, :, 1::2] += y_coords
        anchors[:, y, :, 1::2] /= grid_size[1].float()

        anchors = anchors.reshape(-1, 4)

        return anchors

    @staticmethod
    def validate_anchors(anchors):
        out_of_bounds = torch.nonzero(anchors[:, 0] < 0)
        anchors[out_of_bounds] = -1
        out_of_bounds = torch.nonzero(anchors[:, 1] < 0)
        anchors[out_of_bounds] = -1
        out_of_bounds = torch.nonzero(anchors[:, 2] > 1)
        anchors[out_of_bounds] = -1
        out_of_bounds = torch.nonzero(anchors[:, 3] > 1)
        anchors[out_of_bounds] = -1
        keep = torch.nonzero(anchors[:, 0] >= 0)

        return keep.squeeze()

    def training(self, train_data, backbone, optimizer, batch_size=64, epochs=1,
                 val_data=None, shuffle=True, multi_scale=True, checkpoint_frequency=100):

        self.train()

        train_dataloader = DataLoader(dataset=train_data,
                                      batch_size=1,
                                      shuffle=shuffle,
                                      num_workers=NUM_WORKERS)

        train_loss = []
        val_loss = []

        for epoch in range(1, epochs + 1):
            image_loss = []
            if multi_scale:
                random_size = np.random.randint(49, 52) * NETWORK_STRIDE
                train_data.set_image_size(random_size, random_size)
            with tqdm(total=len(train_dataloader),
                      desc='Epoch: [{}/{}]'.format(epoch, epochs),
                      leave=True,
                      unit='batches') as inner:
                for image, image_info, image_transforms in train_dataloader:
                    image = image.to(self.device)
                    image_info_ = index_dict_list(image_info, 0)
                    image_transforms_ = index_dict_list(image_transforms, 0)
                    target_bboxes, _ = train_data.get_gt_bboxes(image_info_, image_transforms_)
                    target_bboxes = target_bboxes.to(self.device)
                    if target_bboxes.numel() > 0:
                        optimizer.zero_grad()
                        features = backbone(image)
                        grid_size = torch.tensor(features.shape[-2:], device=self.device)
                        anchors = self.construct_anchors(grid_size)
                        cls, reg = self(features)
                        valid = self.validate_anchors(anchors)
                        cls = cls[0][valid]
                        reg = reg[0][valid]
                        anchors = anchors[valid]
                        reg = deparameterize_bboxes(reg, anchors)
                        ious = jaccard(anchors, target_bboxes)
                        max_iou, argmax_iou = torch.max(ious, dim=1)
                        positive_mask = max_iou > RPN_HI_THRESHOLD
                        abs_max_iou = torch.argmax(ious, dim=0)
                        positive_mask[abs_max_iou] = 1
                        positive_mask = torch.nonzero(positive_mask)
                        negative_mask = torch.nonzero(max_iou < RPN_LO_THRESHOLD)
                        target_anchors = random_choice(negative_mask, batch_size)
                        rpn_pos_numel = int(batch_size * RPN_POS_RATIO)
                        if positive_mask.numel() < rpn_pos_numel:
                            rpn_pos_numel = positive_mask.numel()
                        target_anchors[:rpn_pos_numel] = random_choice(positive_mask, rpn_pos_numel)
                        target_cls = torch.zeros(batch_size, 2, device=self.device)
                        target_cls[:rpn_pos_numel, 0] = 1
                        target_cls[rpn_pos_numel:, 1] = 1
                        target_bboxes = target_bboxes[argmax_iou][target_anchors]
                        target_reg = parameterize_bboxes(target_bboxes, anchors[target_anchors])
                        cls = cls[target_anchors]
                        reg = parameterize_bboxes(reg[target_anchors], anchors[target_anchors])
                        loss = self.loss(cls, reg, target_cls, target_reg)
                        image_loss.append(loss['total'].item())
                        loss['total'].backward()
                        optimizer.step()
                        inner.set_postfix_str(' RPN Training Loss: {:.6f}'.format(np.mean(image_loss)))
                    inner.update()
                train_loss.append(np.mean(image_loss))
                if val_data is not None:
                    val_data.reset_image_size()
                    val_loss.append(self.calculate_loss(val_data, backbone, batch_size))
                    inner.set_postfix_str(' RPN Training Loss: {:.6f},  RPN Validation Loss: {:.6f}'.format(train_loss[-1],
                                                                                                    val_loss[-1]))
                else:
                    inner.set_postfix_str(' RPN Training Loss: {:.6f}'.format(train_loss[-1]))

        return train_loss, val_loss

    @staticmethod
    def loss(cls, reg, target_cls, target_reg, focal=True):

        loss = dict()

        if focal:
            loss['cls'] = FocalLoss(reduction='mean')(cls, target_cls)
        else:
            loss['cls'] = nn.BCELoss(reduction='mean')(cls, target_cls)

        obj_mask = torch.nonzero(target_cls[:, 0]).squeeze()
        lambd = 1. / cls.shape[0]
        loss['reg'] = lambd * nn.SmoothL1Loss(reduction='sum')(reg[obj_mask],
                                                               target_reg[obj_mask])

        loss['total'] = loss['cls'] + loss['reg']
        return loss

    def calculate_loss(self, data, backbone, batch_size=256, fraction=0.05):
        """
        Calculates the loss for a random partition of a given dataset without
        tracking gradients. Useful for displaying the validation loss during
        training.
        Parameters
        ----------
        data : PascalDatasetImage
            A dataset object which returns images and image info to use for calculating
            the loss. Only a fraction of the images in the dataset will be tested.
        backbone : nn.Module
            A CNN that is used to convert an image to an n-dimensional feature map that
            can be processed by the RPN.
        batch_size : int
            The number of ROIs per image for which the loss should be calculated. This
            should not influence the value that the function returns, but will affect
            performance.
        fraction : float
            The fraction of images from data that the loss should be calculated for.

        Returns
        -------
        float
            The mean loss over the fraction of the images that were sampled from
            the data.
        """
        val_dataloader = DataLoader(dataset=data,
                                    batch_size=1,
                                    shuffle=True,
                                    num_workers=NUM_WORKERS)
        losses = []
        with torch.no_grad():
            for i, (image, image_info, image_transforms) in enumerate(val_dataloader, 1):
                image = image.to(self.device)
                image_info_ = index_dict_list(image_info, 0)
                image_transforms_ = index_dict_list(image_transforms, 0)
                target_bboxes, _ = data.get_gt_bboxes(image_info_, image_transforms_)
                target_bboxes = target_bboxes.to(self.device)
                features = backbone(image)
                grid_size = torch.tensor(features.shape[-2:], device=self.device)
                anchors = self.construct_anchors(grid_size)
                cls, reg = self(features, nms=False)
                valid = self.validate_anchors(anchors)
                cls = cls[valid]
                reg = reg[valid]
                anchors = anchors[valid]
                reg = parameterize_bboxes(reg, anchors)
                ious = jaccard(anchors, target_bboxes)
                max_iou, argmax_iou = torch.max(ious, dim=1)
                positive_mask = max_iou > RPN_HI_THRESHOLD
                abs_max_iou = torch.argmax(ious, dim=0)
                positive_mask[abs_max_iou] = 1
                positive_mask = torch.nonzero(positive_mask)
                negative_mask = torch.nonzero(max_iou < RPN_LO_THRESHOLD)
                target_anchors = random_choice(negative_mask, batch_size)
                rpn_pos_numel = int(batch_size * RPN_POS_RATIO)
                if positive_mask.numel() < rpn_pos_numel:
                    rpn_pos_numel = positive_mask.numel()
                target_anchors[:rpn_pos_numel] = random_choice(positive_mask, rpn_pos_numel)
                target_cls = torch.zeros(batch_size, 2, device=self.device)
                target_cls[:rpn_pos_numel, 0] = 1
                target_cls[rpn_pos_numel:, 1] = 1
                target_bboxes = target_bboxes[argmax_iou][target_anchors]
                target_reg = parameterize_bboxes(target_bboxes, anchors[target_anchors])
                cls = cls[target_anchors]
                reg = reg[target_anchors]
                loss = self.loss(cls, reg, target_cls, target_reg)
                losses.append(loss['total'].item())
                if i > len(val_dataloader) * fraction:
                    break

        return np.mean(losses)


class FasterRCNN(nn.Module):

    def __init__(self, anchors, name='FasterR-CNN', num_classes=20, model_path='models/vgg11_bn-6002323d.pth', device='cuda'):
        super(FasterRCNN, self).__init__()

        self.model_path = model_path
        self.fast_rcnn = FastRCNN(num_classes=num_classes,
                                  pretrained=True,
                                  model_path=model_path,
                                  device=device)

        self.rpn = RPN(anchors=anchors,
                       device=device)

        self.name = name
        self.num_classes = num_classes
        self.device = device

    def forward(self, x):
        features = self.fast_rcnn.features(x)
        classes, reg = self.rpn(features)
        grid_size = torch.tensor(features.shape[-2:], device=self.device)
        classes, rois = self.rpn.post_process(classes, reg, grid_size, max_rois=MAX_TEST_RPN_ROIS, nms=True)
        cls, reg = self.fast_rcnn(features, rois, extract_features=False)
        cls, bboxes = self.fast_rcnn.post_process(cls, reg, rois)

        return cls, bboxes

    def predict(self, dataset, batch_size=10, confidence_threshold=0.1, overlap_threshold=0.5, show=True, export=True):

        self.eval()

        dataloader = DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=NUM_WORKERS)

        image_idx_ = []
        bboxes_ = []
        confidence_ = []
        classes_ = []

        context = tqdm(total=len(dataloader), desc='Exporting', leave=True) if export else nullcontext()

        with torch.no_grad():
            with context as pbar:
                for images, image_info, targets in dataloader:
                    images = images.to(self.device)
                    cls, bboxes = self(images)
                    (bboxes,
                     classes,
                     confidences,
                     image_idx) = self.fast_rcnn.process_bboxes(cls, bboxes,
                                                                confidence_threshold=confidence_threshold,
                                                                overlap_threshold=overlap_threshold)
                    if show:
                        for idx, image in enumerate(images):
                            width = image_info['width'][idx]
                            height = image_info['height'][idx]
                            image = to_numpy_image(image, size=(width, height))
                            mask = image_idx == idx
                            for bbox, cls, confidence in zip(bboxes[mask], classes[mask], confidences[mask]):
                                name = dataset.classes[cls]
                                add_bbox_to_image(image, bbox, confidence, name)
                            plt.imshow(image)
                            plt.show()

                    if export:
                        for idx in range(len(images)):
                            mask = image_idx == idx
                            for bbox, cls, confidence in zip(bboxes[mask], classes[mask], confidences[mask]):
                                name = dataset.classes[cls]
                                ids = image_info['id'][idx]
                                set_name = image_info['dataset'][idx]
                                confidence = confidence.item()
                                x1, y1, x2, y2 = bbox.detach().cpu().numpy()
                                width = image_info['width'][idx].item()
                                height = image_info['height'][idx].item()
                                x1 *= width
                                x2 *= width
                                y1 *= height
                                y2 *= height
                                export_prediction(cls=name,
                                                  image_id=ids,
                                                  left=x1,
                                                  top=y1,
                                                  right=x2,
                                                  bottom=y2,
                                                  confidence=confidence,
                                                  set_name=set_name)
                        pbar.update()

                    bboxes_.append(bboxes)
                    confidence_.append(confidences)
                    classes_.append(classes)
                    image_idx_.append(image_idx)

            if len(bboxes_) > 0:
                bboxes = torch.cat(bboxes_).view(-1, 4)
                classes = torch.cat(classes_).flatten()
                confidence = torch.cat(confidence_).flatten()
                image_idx = torch.cat(image_idx_).flatten()

                return bboxes, classes, confidence, image_idx
            else:
                return torch.tensor([], device=self.device), \
                       torch.tensor([], device=self.device), \
                       torch.tensor([], device=self.device), \
                       torch.tensor([], device=self.device)

    def alternate_training(self, train_data, image_batch_size=2, roi_batch_size=64,
                           epochs=1, lr=1e-4, momentum=0.99, val_data=None, shuffle=True,
                           multi_scale=True, checkpoint_frequency=1, stage=None):

        if stage is None:
            assert isinstance(image_batch_size, (list, tuple))
            assert isinstance(roi_batch_size, (list, tuple))
            assert isinstance(epochs, (list, tuple))
            assert len(image_batch_size) == 4
            assert len(roi_batch_size) == 4
            assert len(epochs) == 4

            train_loss = []
            val_loss = []

            for i in range(4):
                loss = self.alternate_training(train_data=train_data,
                                               val_data=val_data,
                                               epochs=epochs[i],
                                               image_batch_size=image_batch_size[i],
                                               roi_batch_size=roi_batch_size[i],
                                               multi_scale=multi_scale,
                                               shuffle=shuffle,
                                               stage=i)
                train_loss.append(loss[0])
                val_loss.append(loss[1])

            return train_loss, val_loss

        elif stage == 0:

            plist = [{'params': self.fast_rcnn.features.parameters()},
                     {'params': self.rpn.parameters()}
                     ]
            optimizer = optim.SGD(plist, lr=lr, momentum=momentum, weight_decay=5e-4)

            self.fast_rcnn.train()
            train_loss, val_loss = self.rpn.training(train_data=train_data,
                                                     backbone=self.fast_rcnn.features,
                                                     optimizer=optimizer,
                                                     batch_size=roi_batch_size,
                                                     epochs=epochs,
                                                     val_data=val_data,
                                                     shuffle=shuffle,
                                                     multi_scale=multi_scale,
                                                     checkpoint_frequency=checkpoint_frequency)

            self.save_model(self.name + '_{}.pkl'.format('debug_stage_0'))

            return train_loss, val_loss

        elif stage == 1:

            backbone = copy.deepcopy(self.fast_rcnn.features)
            self.fast_rcnn = FastRCNN(num_classes=self.num_classes,
                                      pretrained=True,
                                      model_path=self.model_path,
                                      device=self.device)
            plist = [{'params': self.fast_rcnn.parameters()}]
            optimizer = optim.SGD(plist, lr=lr, momentum=momentum, weight_decay=5e-4)

            self.rpn.eval()
            train_loss, val_loss = self.fast_rcnn.training(train_data=train_data,
                                                           optimizer=optimizer,
                                                           backbone=backbone,
                                                           rpn=self.rpn,
                                                           max_rois=MAX_TRAIN_RPN_ROIS,
                                                           image_batch_size=image_batch_size,
                                                           roi_batch_size=roi_batch_size,
                                                           epochs=epochs,
                                                           val_data=val_data,
                                                           shuffle=shuffle,
                                                           multi_scale=multi_scale)

            self.save_model(self.name + '_{}.pkl'.format('debug_stage_1'))

            return train_loss, val_loss

        elif stage == 2:

            plist = [{'params': self.rpn.parameters()}]
            optimizer = optim.SGD(plist, lr=lr, momentum=momentum, weight_decay=5e-4)

            self.fast_rcnn.eval()
            self.fast_rcnn.freeze()
            train_loss, val_loss = self.rpn.training(train_data=train_data,
                                                     backbone=self.fast_rcnn.features,
                                                     optimizer=optimizer,
                                                     batch_size=roi_batch_size,
                                                     epochs=epochs,
                                                     val_data=val_data,
                                                     shuffle=shuffle,
                                                     multi_scale=multi_scale,
                                                     checkpoint_frequency=checkpoint_frequency)

            self.save_model(self.name + '_{}.pkl'.format('debug_stage_2'))

            return train_loss, val_loss

        elif stage == 3:

            plist = [{'params': self.fast_rcnn.classifier.parameters()},
                     {'params': self.fast_rcnn.cls.parameters()},
                     {'params': self.fast_rcnn.reg.parameters()}]
            optimizer = optim.SGD(plist, lr=lr, momentum=momentum, weight_decay=5e-4)

            self.rpn.eval()
            self.fast_rcnn.freeze()
            train_loss, val_loss = self.fast_rcnn.training(train_data=train_data,
                                                           optimizer=optimizer,
                                                           backbone=self.fast_rcnn.features,
                                                           rpn=self.rpn,
                                                           max_rois=MAX_TRAIN_RPN_ROIS,
                                                           image_batch_size=image_batch_size,
                                                           roi_batch_size=roi_batch_size,
                                                           epochs=epochs,
                                                           val_data=val_data,
                                                           shuffle=shuffle,
                                                           multi_scale=multi_scale)

            self.save_model(self.name + '_{}.pkl'.format('debug_stage_3'))

            return train_loss, val_loss

    def joint_training(self, train_data, optimizer, max_rois=2000, image_batch_size=2,
                       roi_batch_size=32, epochs=1, val_data=None, shuffle=True, multi_scale=True, checkpoint_frequency=100):

        self.train()

        train_dataloader = DataLoader(dataset=train_data,
                                      batch_size=image_batch_size,
                                      shuffle=shuffle,
                                      num_workers=NUM_WORKERS)

        train_loss = []
        val_loss = []

        for epoch in range(1, epochs + 1):
            rpn_loss = []
            frcn_loss = []
            if multi_scale:
                random_size = np.random.randint(49, 52) * NETWORK_STRIDE
                train_data.set_image_size(random_size, random_size)
            with tqdm(total=len(train_dataloader),
                      desc='Epoch: [{}/{}]'.format(epoch, epochs),
                      leave=True,
                      unit='batches') as inner:
                for images, images_info, images_transforms in train_dataloader:
                    images = images.to(self.device)
                    image_batch_size = len(images)
                    frcn_target_rois = [torch.zeros(roi_batch_size, 4, device=self.device)] * image_batch_size
                    frcn_target_classes = torch.zeros(image_batch_size, roi_batch_size, self.num_classes + 1,
                                                      device=self.device)
                    target_bboxes = torch.zeros(image_batch_size, roi_batch_size, 4, device=self.device)
                    optimizer.zero_grad()
                    features = self.fast_rcnn.features(images)
                    for i in range(image_batch_size):
                        image_info_ = index_dict_list(images_info, i)
                        image_transforms_ = index_dict_list(images_transforms, i)
                        image_bboxes, image_classes = train_data.get_gt_bboxes(image_info_, image_transforms_)
                        image_bboxes = image_bboxes.to(self.device)
                        image_classes = image_classes.to(self.device)
                        if image_bboxes.numel() > 0:
                            cls, reg = self.rpn(features[i][None])
                            cls_ = cls.clone()
                            reg_ = reg.clone()
                            grid_size = torch.tensor(features.shape[-2:], device=self.device)
                            anchors = self.rpn.construct_anchors(grid_size)
                            valid = self.rpn.validate_anchors(anchors)
                            cls = cls[0][valid]
                            reg = reg[0][valid]
                            anchors = anchors[valid]
                            reg = deparameterize_bboxes(reg, anchors)
                            ious = jaccard(anchors, image_bboxes)
                            max_iou, argmax_iou = torch.max(ious, dim=1)
                            positive_mask = max_iou > RPN_HI_THRESHOLD
                            abs_max_iou = torch.argmax(ious, dim=0)
                            positive_mask[abs_max_iou] = 1
                            positive_mask = torch.nonzero(positive_mask)
                            negative_mask = torch.nonzero(max_iou < RPN_LO_THRESHOLD)
                            target_anchors = random_choice(negative_mask, roi_batch_size)
                            rpn_pos_numel = int(roi_batch_size * RPN_POS_RATIO)
                            if positive_mask.numel() < rpn_pos_numel:
                                rpn_pos_numel = positive_mask.numel()
                            target_anchors[:rpn_pos_numel] = random_choice(positive_mask, rpn_pos_numel)
                            target_cls = torch.zeros(roi_batch_size, 2, device=self.device)
                            target_cls[:rpn_pos_numel, 0] = 1
                            target_cls[rpn_pos_numel:, 1] = 1
                            image_bboxes_ = image_bboxes[argmax_iou][target_anchors]
                            target_reg = parameterize_bboxes(image_bboxes_, anchors[target_anchors])
                            cls = cls[target_anchors]
                            reg = parameterize_bboxes(reg[target_anchors], anchors[target_anchors])
                            loss = self.rpn.loss(cls, reg, target_cls, target_reg)
                            rpn_loss.append(loss['total'].item())
                            loss['total'].backward(retain_graph=True)
                            # FRCN
                            rois = self.rpn.post_process(cls_, reg_, grid_size, max_rois, True)[1][0]
                            ious = jaccard(rois, image_bboxes)
                            max_iou, argmax_iou = torch.max(ious, dim=1)
                            positive_mask = max_iou > FRCN_HI_THRESHOLD
                            abs_max_iou = torch.argmax(ious, dim=0)
                            positive_mask[abs_max_iou] = 1  # Added ROI with maximum IoU to positive samples.
                            positive_mask = torch.nonzero(positive_mask)
                            negative_mask = torch.nonzero(
                                (FRCN_LO_LO_THRESHOLD < max_iou) * (max_iou < FRCN_LO_HI_THRESHOLD))
                            if negative_mask.numel() > 0:
                                samples = random_choice(negative_mask, roi_batch_size, replace=True)
                            else:
                                negative_mask = torch.nonzero(max_iou < FRCN_LO_HI_THRESHOLD)
                                if negative_mask.numel() > 0:
                                    samples = random_choice(negative_mask, roi_batch_size, replace=True)
                                else:
                                    negative_mask = torch.nonzero(max_iou < FRCN_HI_THRESHOLD)
                                    samples = random_choice(negative_mask, roi_batch_size, replace=True)
                            frcn_pos_numel = int(roi_batch_size * FRCN_POS_RATIO)
                            if positive_mask.numel() < frcn_pos_numel:
                                frcn_pos_numel = positive_mask.numel()
                            if frcn_pos_numel > 0:
                                samples[:frcn_pos_numel] = random_choice(positive_mask, frcn_pos_numel)
                            frcn_target_rois[i] = rois[samples].detach()
                            frcn_target_classes[i][:frcn_pos_numel] = image_classes[argmax_iou[samples[:frcn_pos_numel]]]
                            frcn_target_classes[i][frcn_pos_numel:, 0] = 1.
                            target_bboxes[i] = parameterize_bboxes(image_bboxes[argmax_iou[samples]], rois[samples])
                    cls, reg = self.fast_rcnn(images, frcn_target_rois)
                    loss = self.fast_rcnn.loss(cls, reg, frcn_target_classes.detach(), target_bboxes.detach())
                    frcn_loss.append(loss['total'].item())
                    loss['total'].backward()
                    optimizer.step()
                    inner.set_postfix_str(' RPN Training Loss: {:.6f},  FRCN Training Loss: {:.6f}'.format(np.mean(rpn_loss), np.mean(frcn_loss)))
                    inner.update()
                train_loss.append([np.mean(rpn_loss), np.mean(frcn_loss)])
                if val_data is not None:
                    val_data.reset_image_size()
                    val_loss.append(self.calculate_loss(val_data, backbone, batch_size))
                    inner.set_postfix_str(' RPN Training Loss: {:.6f},  '
                                          'FRCN Training Loss: {:.6f},  '
                                          'RPN Validation Loss: {:.6f},  '
                                          'FRCN Validation Loss: {:.6f}'.format(train_loss[-1][0],
                                                                                train_loss[-1][1],
                                                                                val_loss[-1][0],
                                                                                val_loss[-1][1]))
                else:
                    inner.set_postfix_str(' RPN Training Loss: {:.6f},  FRCN Training Loss: {:.6f}'.format(train_loss[-1][0], train_loss[-1][1]))

        self.save_model(self.name + '_{}.pkl'.format('joint_training'))

        return train_loss, val_loss

    def save_model(self, name):
        """
        Save the entire Faster R-CNN model by using the built-in Python
        pickle module.
        Parameters
        ----------
        name
            The filename where the model should be saved.
        """
        pickle.dump(self, open(name, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
