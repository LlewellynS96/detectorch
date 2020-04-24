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
from utils import jaccard, sample_ids, parameterize_bboxes, deparameterize_bboxes, access_dict_list, export_prediction
from utils import to_numpy_image, add_bbox_to_image, nullcontext
from utils import NUM_WORKERS
from layers import FocalLoss


NETWORK_STRIDE = 16
RPN_HI_THRESHOLD = 0.7
RPN_LO_THRESHOLD = 0.3
FRCN_HI_THRESHOLD = 0.5
FRCN_LO_LO_THRESHOLD = 0.01
FRCN_LO_HI_THRESHOLD = 0.5
RPN_POS_RATIO = 0.5
FRCN_POS_RATIO = 0.25
RPN_NMS_THRESHOLD = 0.7
MAX_TRAIN_RPN_ROIS = 2000
MAX_TEST_RPN_ROIS = 300


class VGGBackbone(nn.Module):
    def __init__(self,
                 model=models.vgg16(),
                 model_path='models/vgg16-397923af.pth',
                 pretrained=True,
                 device='cuda'):
        super(VGGBackbone, self).__init__()
        self.device = device
        self.model = model
        if pretrained:
            self.model_path = model_path
            print('Loading pretrained weights from {}'.format(self.model_path))
            state_dict = torch.load(self.model_path)
            self.model.load_state_dict({k: v for k, v in state_dict.items() if k in self.model.state_dict()})
        self.model.features = nn.Sequential(*list(self.model.features.children())[:-1])
        self.model.classifier = nn.Sequential(*list(self.model.classifier.children())[:-2])

        self.features = self.model.features
        self.classifier = self.model.classifier

        self.out_features = 4096

        self.to(device)


class FastRCNN(nn.Module):

    def __init__(self,
                 num_classes=20,
                 backbone=None,
                 device='cuda'):
        super(FastRCNN, self).__init__()
        if backbone is None:
            backbone = VGGBackbone()
        self.num_classes = num_classes
        self.device = device
        self.backbone = backbone
        self.reg = nn.Linear(backbone.out_features, self.num_classes * 4)
        self.cls = nn.Linear(backbone.out_features, self.num_classes + 1)

        self.to(device)

    def forward(self, x, rois, extract_features=True):
        assert x.shape[0] == 1
        if extract_features:
            features = self.backbone.features(x)
        else:
            features = x
        grid_size = features.shape[-2:]
        rois_idxs = rois.clone()
        rois_idxs[:, ::2] *= grid_size[1]
        rois_idxs[:, 1::2] *= grid_size[0]
        roi_features = ops.roi_pool(features, [rois_idxs], output_size=(7, 7))
        # roi_features = ops.roi_align(features, [rois_idxs], output_size=(7, 7))
        classifier = self.backbone.classifier(roi_features.view(1, -1, 512 * 7 * 7))
        reg = self.reg(classifier)
        cls = self.cls(classifier)

        reg = reg.reshape(1, -1, self.num_classes, 4)

        return reg, cls

    def freeze(self):
        """
            Freezes the backbone (conv and bn) of the VGG network.
        """
        for param in self.backbone.features.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """
            Freezes the backbone (conv and bn) of the VGG network.
        """
        for param in self.backbone.features.parameters():
            param.requires_grad = True

    def mini_freeze(self, num_layers=12):
        """
            Freezes the first few layers of the backbone (conv and bn) of the VGG network.
        """
        for param in list(self.backbone.features.parameters())[:num_layers]:
            param.requires_grad = False

    def fit(self,
            train_data,
            optimizer,
            scheduler=None,
            alt_feature_extractor=None,
            rpn=None,
            rois=None,
            roi_batch_size=64,
            epochs=1,
            val_data=None,
            shuffle=True):
        assert rpn or rois

        self.train()

        train_dataloader = DataLoader(dataset=train_data,
                                      shuffle=shuffle,
                                      num_workers=NUM_WORKERS)

        train_loss = []
        val_loss = []

        for epoch in range(1, epochs + 1):
            batch_loss = []
            with tqdm(total=len(train_dataloader),
                      desc='Epoch: [{}/{}]'.format(epoch, epochs),
                      leave=True,
                      unit='batches') as inner:
                for images, images_info, (bboxes, classes) in train_dataloader:
                    images = images.to(self.device)
                    bboxes = bboxes.to(self.device)
                    classes = classes.to(self.device)
                    target_rois = torch.zeros(roi_batch_size, 4, device=self.device)
                    target_classes = torch.zeros(roi_batch_size, self.num_classes + 1, device=self.device)
                    target_bboxes = torch.zeros(roi_batch_size, 4, device=self.device)
                    with torch.no_grad():
                        if alt_feature_extractor is None:
                            features = self.features(images)
                        else:
                            features = alt_feature_extractor(images)
                    if bboxes.numel() > 0:
                        if rpn is not None:
                            cls, reg = rpn(features)
                            grid_size = torch.tensor(features.shape[-2:], device=self.device)
                            rois = rpn.post_process(cls, reg, grid_size, MAX_TRAIN_RPN_ROIS, True)[1][0]
                        ious = jaccard(rois, bboxes)
                        max_iou, argmax_iou = torch.max(ious, dim=1)
                        pos_mask = max_iou > FRCN_HI_THRESHOLD
                        abs_max_iou = torch.argmax(ious, dim=0)
                        pos_mask[abs_max_iou] = 1  # Added ROI with maximum IoU to pos samples.
                        pos_mask = torch.nonzero(pos_mask)
                        neg_mask = torch.nonzero((FRCN_LO_LO_THRESHOLD < max_iou) *
                                                      (max_iou < FRCN_LO_HI_THRESHOLD))
                        if neg_mask.numel() > 0:
                            samples = random_choice(neg_mask, roi_batch_size, replace=True)
                        else:
                            neg_mask = torch.nonzero(max_iou < FRCN_LO_HI_THRESHOLD)
                            if neg_mask.numel() > 0:
                                samples = random_choice(neg_mask, roi_batch_size, replace=True)
                            else:
                                neg_mask = torch.nonzero(max_iou < FRCN_HI_THRESHOLD)
                                samples = random_choice(neg_mask, roi_batch_size, replace=True)
                        frcn_pos_numel = int(roi_batch_size * FRCN_POS_RATIO)
                        if pos_mask.numel() < frcn_pos_numel:
                            frcn_pos_numel = pos_mask.numel()
                        if frcn_pos_numel > 0:
                            samples[:frcn_pos_numel] = random_choice(pos_mask, frcn_pos_numel)
                        target_rois[i] = rois[samples]
                        target_classes[i][:frcn_pos_numel] = image_classes[argmax_iou[samples[:frcn_pos_numel]]]
                        target_classes[i][frcn_pos_numel:, 0] = 1.
                        target_bboxes[i] = parameterize_bboxes(image_bboxes[argmax_iou[samples]], rois[samples])
                    optimizer.zero_grad()
                    cls, reg = self(images, target_rois)
                    loss = self.loss(cls, reg, target_classes.detach(), target_bboxes.detach())
                    batch_loss.append(loss['total'].item())
                    loss['total'].backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5., norm_type='inf')
                    optimizer.step()
                    inner.set_postfix_str(' FRCN Training Loss: {:.6f}'.format(np.mean(batch_loss)))
                    inner.update()
                train_loss.append(np.mean(batch_loss))
                if val_data is not None:
                    val_data.reset_image_size()
                    val_loss.append(self.calculate_loss(val_data, backbone, rpn, 128))
                    inner.set_postfix_str(' FRCN Training Loss: {:.6f},  FRCN Validation Loss: {:.6f}'
                                          .format(train_loss[-1], val_loss[-1]))
                else:
                    inner.set_postfix_str(' FRCN Training Loss: {:.6f}'.format(train_loss[-1]))
                with open('loss_frcn.txt', 'a') as fl:
                    fl.writelines('Epoch {}: FRCN Training Loss: {:.6f}\n'.format(epoch,
                                                                                  train_loss[-1]))
            if scheduler is not None:
                scheduler.step()

        return train_loss, val_loss

    def loss(self, reg, cls, target_reg, target_cls, stats, focal=False):

        loss = dict()
        lambd = 1. / cls.shape[0] / cls.shape[1]
        target = torch.argmax(target_cls, dim=-1).flatten()

        if focal:
            loss['cls'] = lambd * FocalLoss(reduction='sum', gamma=1.0)(cls.view(-1, self.num_classes + 1), target)
        else:
            loss['cls'] = lambd * nn.CrossEntropyLoss(reduction='sum')(cls.view(-1, self.num_classes + 1), target)

        lambd = 5. / cls.shape[0] / cls.shape[1]

        obj_mask = torch.nonzero(target_cls[:, :, 0] == 0)
        arange = torch.arange(len(obj_mask))
        obj_mask = tuple(obj_mask.t())
        if len(obj_mask) > 0:
            cls = torch.argmax(target_cls, dim=-1) - 1
            loss['reg'] = 4. * lambd * nn.SmoothL1Loss(reduction='sum')(reg[obj_mask][arange, cls[obj_mask]],
                                                                        target_reg[obj_mask])
        else:
            loss['reg'] = 0.

        loss['total'] = loss['cls'] + loss['reg']

        return loss

    def calculate_loss(self, data, backbone, rpn, roi_batch_size=256, fraction=0.01):
        """
        Calculates the loss for a random partition of a given dataset without
        tracking gradients. Useful for displaying the validation loss during
        training or the test loss during evaluation.
        Parameters
        ----------
        data : PascalDatasetImage
            A dataset object which returns images and image info to use for calculating
            the loss. Only a fraction of the images in the dataset will be tested.
        backbone : nn.Module
            A CNN that is used to convert an image to an n-dimensional feature map that
            can be processed by the RPN.
        roi_batch_size : int
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
                                    batch_size=2,
                                    shuffle=False,
                                    num_workers=NUM_WORKERS)
        losses = []
        with torch.no_grad():
            for i, (images, images_info, images_transforms) in enumerate(val_dataloader, 1):
                images = images.to(self.device)
                image_batch_size = len(images)
                target_rois = [torch.zeros(roi_batch_size, 4, device=self.device)] * image_batch_size
                target_classes = torch.zeros(image_batch_size, roi_batch_size, self.num_classes + 1, device=self.device)
                target_bboxes = torch.zeros(image_batch_size, roi_batch_size, 4, device=self.device)
                if backbone is None:
                    features = self.features(images)
                else:
                    features = backbone(images)
                for j in range(image_batch_size):
                    image_info_copy = access_dict_list(images_info, j)
                    image_transforms_copy = access_dict_list(images_transforms, j)
                    image_bboxes, image_classes = data.get_gt_bboxes(image_info_copy, image_transforms_copy)
                    image_bboxes = image_bboxes.to(self.device)
                    image_classes = image_classes.to(self.device)
                    if image_bboxes.numel() > 0:
                        if rpn is not None:
                            cls, reg = rpn(features[j][None])
                            grid_size = torch.tensor(features.shape[-2:], device=self.device)
                            rois = rpn.post_process(cls, reg, grid_size, MAX_TRAIN_RPN_ROIS, True)[1][0]
                        ious = jaccard(rois, image_bboxes)
                        max_iou, argmax_iou = torch.max(ious, dim=1)
                        pos_mask = max_iou > FRCN_HI_THRESHOLD
                        abs_max_iou = torch.argmax(ious, dim=0)
                        pos_mask[abs_max_iou] = 1  # Added ROI with maximum IoU to pos samples.
                        pos_mask = torch.nonzero(pos_mask)
                        neg_mask = torch.nonzero((FRCN_LO_LO_THRESHOLD < max_iou) *
                                                      (max_iou < FRCN_LO_HI_THRESHOLD))
                        if neg_mask.numel() > 0:
                            samples = random_choice(neg_mask, roi_batch_size, replace=True)
                        else:
                            neg_mask = torch.nonzero(max_iou < FRCN_LO_HI_THRESHOLD)
                            if neg_mask.numel() > 0:
                                samples = random_choice(neg_mask, roi_batch_size, replace=True)
                            else:
                                neg_mask = torch.nonzero(max_iou < FRCN_HI_THRESHOLD)
                                samples = random_choice(neg_mask, roi_batch_size, replace=True)
                        frcn_pos_numel = int(roi_batch_size * FRCN_POS_RATIO)
                        if pos_mask.numel() < frcn_pos_numel:
                            frcn_pos_numel = pos_mask.numel()
                        if frcn_pos_numel > 0:
                            samples[:frcn_pos_numel] = random_choice(pos_mask, frcn_pos_numel)
                        target_rois[j] = rois[samples]
                        target_classes[j][:frcn_pos_numel] = image_classes[argmax_iou[samples[:frcn_pos_numel]]]
                        target_classes[j][frcn_pos_numel:, 0] = 1.
                        target_bboxes[j] = parameterize_bboxes(image_bboxes[argmax_iou[samples]], rois[samples])
                cls, reg = self(images, target_rois)
                loss = self.loss(cls, reg, target_classes.detach(), target_bboxes.detach())
                losses.append(loss['total'].item())
                if i > len(val_dataloader) * fraction:
                    break

        return np.mean(losses)

    def post_process(self, reg, cls, rois):
        assert reg.shape[0] == 1
        assert cls.shape[0] == 1

        reg = reg[0]
        cls = cls[0]

        cls = F.softmax(cls, dim=-1)

        classes = torch.argmax(cls[:, 1:], dim=-1)
        arange = torch.arange(len(classes))
        bboxes = reg[arange, classes]

        if bboxes.numel() > 0:
            bboxes = deparameterize_bboxes(bboxes, rois)
            bboxes = torch.clamp(bboxes, min=0., max=1.)

            cls[:, 0] = 0.
            sort = torch.argsort(cls[arange, classes + 1], descending=True)
            classes = cls[sort]
            bboxes = bboxes[sort]
        else:
            classes.append(torch.tensor([], device=self.device))
            bboxes.append(torch.tensor([], device=self.device))

        return bboxes, classes

    def process_bboxes(self, bboxes, classes, image_size, confidence_threshold=0.01, overlap_threshold=0.6, nms=True):
        if classes.numel() == 0:
            return torch.tensor([], device=self.device), \
                   torch.tensor([], device=self.device), \
                   torch.tensor([], device=self.device)

        confidence, classes = torch.max(classes, dim=-1)

        mask = confidence > confidence_threshold

        if sum(mask) == 0:
            return torch.tensor([], device=self.device), \
                   torch.tensor([], device=self.device), \
                   torch.tensor([], device=self.device)

        bboxes = bboxes[mask]
        confidence = confidence[mask]
        classes = classes[mask]

        if nms:
            bboxes_tmp = []
            classes_tmp = []
            confidence_tmp = []

            cls = torch.unique(classes)
            for c in cls:
                cls_mask = (classes == c).nonzero().flatten()
                mask = torchvision.ops.nms(bboxes[cls_mask], confidence[cls_mask], overlap_threshold)
                bboxes_tmp.append(bboxes[cls_mask][mask])
                classes_tmp.append(classes[cls_mask][mask])
                confidence_tmp.append(confidence[cls_mask][mask])

        if len(bboxes) > 0:
            bboxes = torch.cat(bboxes_tmp).view(-1, 4)
            bboxes[:, ::2] *= image_size[0]
            bboxes[:, 1::2] *= image_size[1]
            classes = torch.cat(classes_tmp).flatten()
            confidence = torch.cat(confidence_tmp).flatten()

            return bboxes, classes, confidence
        else:
            return torch.tensor([], device=self.device), \
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
        self.cls.weight.data.normal_(0., 0.01)
        self.cls.bias.data.fill_(0)

        self.reg = nn.Conv2d(in_channels=self.num_features,
                             out_channels=4 * self.num_anchors,
                             kernel_size=1)
        self.reg.weight.data.normal_(0., 0.01)
        self.reg.bias.data.fill_(0)

        self.to(device)

    def forward(self, x):
        assert x.shape[0] == 1

        features = F.relu(self.sliding_windows(x))
        reg = self.reg(features)
        cls = self.cls(features)

        reg = reg.permute(0, 2, 3, 1).reshape(1, -1, 4)
        cls = cls.permute(0, 2, 3, 1).reshape(1, -1, 2)

        return reg, cls

    def post_process(self, reg, cls, grid_size, max_rois=2000, nms=False):
        assert reg.shape[0] == 1
        assert cls.shape[0] == 1

        reg = reg[0]
        cls = cls[0]

        anchors = self.construct_anchors(grid_size)
        classes = F.softmax(cls, dim=-1)
        valid = self.validate_anchors(anchors)
        classes = classes[valid]
        anchors = anchors[valid]

        keep = torch.nonzero(classes[:, 0] > 0.).flatten()  # [REMOVED] Ignore all proposals wit low probabilities
        classes = classes[keep]
        bboxes = deparameterize_bboxes(reg[valid][keep], anchors[keep])
        bboxes = torch.clamp(bboxes, min=0., max=1.)

        keep = np.nonzero(bboxes[:, 2] - bboxes[:, 0]).squeeze()
        bboxes = bboxes[keep]
        classes = classes[keep]

        keep = np.nonzero(bboxes[:, 3] - bboxes[:, 1]).squeeze()
        bboxes = bboxes[keep]
        classes = classes[keep]

        if nms:
            keep = torchvision.ops.nms(bboxes, classes[:, 0], RPN_NMS_THRESHOLD)
            classes = classes[keep]
            bboxes = bboxes[keep]
            classes = classes[:max_rois]
            bboxes = bboxes[:max_rois]

        return bboxes, classes

    def construct_anchors(self, grid_size):

        anchors = torch.zeros((*grid_size, self.num_anchors, 4), device=self.device)

        x = torch.arange(0, grid_size[1], device=self.device)
        y = torch.arange(0, grid_size[0], device=self.device)

        xx, yy = torch.meshgrid(x, y)

        x_coords = x.reshape(1, -1, 1, 1).float()
        y_coords = y.reshape(-1, 1, 1, 1).float()

        anchors[yy, xx, :, :2] = -self.anchors / 2. + .5
        anchors[yy, xx, :, 2:] = self.anchors / 2. + .5

        anchors[:, x, :, ::2] += x_coords
        anchors[:, x, :, ::2] /= grid_size[1].float()
        anchors[y, :, :, 1::2] += y_coords
        anchors[y, :, :, 1::2] /= grid_size[0].float()

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

    def fit(self, train_data, backbone, optimizer, scheduler=None, batch_size=64, epochs=1,
            val_data=None, shuffle=True):

        self.train()

        train_dataloader = DataLoader(dataset=train_data,
                                      batch_size=1,
                                      shuffle=shuffle,
                                      num_workers=NUM_WORKERS)

        train_loss = []
        val_loss = []

        for epoch in range(1, epochs + 1):
            image_loss = []
            with tqdm(total=len(train_dataloader),
                      desc='Epoch: [{}/{}]'.format(epoch, epochs),
                      leave=True,
                      unit='batches') as inner:
                for image, image_info, image_transforms in train_dataloader:
                    image = image.to(self.device)
                    image_info_copy = access_dict_list(image_info, 0)
                    image_transforms_copy = access_dict_list(image_transforms, 0)
                    target_bboxes, _ = train_data.get_gt_bboxes(image_info_copy, image_transforms_copy)
                    target_bboxes = target_bboxes.to(self.device)
                    if target_bboxes.numel() > 0:
                        optimizer.zero_grad()
                        features = backbone(image)
                        grid_size = torch.tensor(features.shape[-2:], device=self.device)
                        anchors = self.construct_anchors(grid_size)
                        reg, cls = self(features)
                        valid = self.validate_anchors(anchors)
                        cls = cls[0][valid]
                        reg = reg[0][valid]
                        anchors = anchors[valid]
                        ious = jaccard(anchors, target_bboxes)
                        max_iou, argmax_iou = torch.max(ious, dim=1)
                        pos_mask = max_iou > RPN_HI_THRESHOLD
                        abs_max_iou = torch.argmax(ious, dim=0)
                        pos_mask[abs_max_iou] = 1
                        pos_mask = torch.nonzero(pos_mask)
                        neg_mask = torch.nonzero(max_iou < RPN_LO_THRESHOLD)
                        target_anchors = random_choice(neg_mask, batch_size)
                        rpn_pos_numel = int(batch_size * RPN_POS_RATIO)
                        if pos_mask.numel() < rpn_pos_numel:
                            rpn_pos_numel = pos_mask.numel()
                        target_anchors[:rpn_pos_numel] = random_choice(pos_mask, rpn_pos_numel)
                        target_cls = torch.zeros(batch_size, 2, device=self.device)
                        target_cls[:rpn_pos_numel, 0] = 1
                        target_cls[rpn_pos_numel:, 1] = 1
                        target_bboxes = target_bboxes[argmax_iou][target_anchors]
                        target_reg = parameterize_bboxes(target_bboxes, anchors[target_anchors])
                        cls = cls[target_anchors]
                        reg = reg[target_anchors]
                        loss = self.loss(reg, cls, target_reg, target_cls)
                        image_loss.append(loss['total'].item())
                        loss['total'].backward()
                        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=.5, norm_type='inf')
                        optimizer.step()
                        inner.set_postfix_str(' RPN Training Loss: {:.6f}'.format(np.mean(image_loss)))
                    inner.update()
                train_loss.append(np.mean(image_loss))
                if val_data is not None:
                    val_data.reset_image_size()
                    val_loss.append(self.calculate_loss(val_data, backbone, batch_size))
                    inner.set_postfix_str(' RPN Training Loss: {:.6f},  RPN Validation Loss: {:.6f}'
                                          .format(train_loss[-1], val_loss[-1]))
                else:
                    inner.set_postfix_str(' RPN Training Loss: {:.6f}'.format(train_loss[-1]))
                with open('loss_rpn.txt', 'a') as fl:
                    fl.writelines('Epoch {}: RPN Training Loss: {:.6f}\n'.format(epoch,
                                                                                 train_loss[-1]))
            if scheduler is not None:
                scheduler.step()

        return train_loss, val_loss

    @staticmethod
    def loss(reg, cls, target_reg, target_cls, stats, focal=False):

        loss = dict()
        target_cls = torch.argmax(target_cls, dim=-1)

        if focal:
            loss['cls'] = FocalLoss(reduction='sum')(cls, target_cls)
        else:
            loss['cls'] = nn.CrossEntropyLoss(reduction='sum')(cls, target_cls)
        loss['cls'] /= cls.shape[0]

        loss['reg'] = nn.SmoothL1Loss(reduction='sum')(reg,
                                                       target_reg)
        loss['reg'] /= cls.shape[0]

        loss['total'] = loss['cls'] + loss['reg']
        return loss

    def calculate_loss(self, data, backbone, batch_size=256, fraction=0.01):
        """
        Calculates the loss for a random partition of a given dataset without
        tracking gradients. Useful for displaying the validation loss during
        training or the test loss during evaluation.
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
                                    shuffle=False,
                                    num_workers=NUM_WORKERS)
        losses = []
        with torch.no_grad():
            for i, (image, image_info, image_transforms) in enumerate(val_dataloader, 1):
                image = image.to(self.device)
                image_info_copy = access_dict_list(image_info, 0)
                image_transforms_copy = access_dict_list(image_transforms, 0)
                target_bboxes, _ = data.get_gt_bboxes(image_info_copy, image_transforms_copy)
                target_bboxes = target_bboxes.to(self.device)
                if target_bboxes.numel() > 0:
                    features = backbone(image)
                    grid_size = torch.tensor(features.shape[-2:], device=self.device)
                    anchors = self.construct_anchors(grid_size)
                    reg, cls = self(features)
                    valid = self.validate_anchors(anchors)
                    cls = cls[0][valid]
                    reg = reg[0][valid]
                    anchors = anchors[valid]
                    reg = deparameterize_bboxes(reg, anchors)
                    ious = jaccard(anchors, target_bboxes)
                    max_iou, argmax_iou = torch.max(ious, dim=1)
                    pos_mask = max_iou > RPN_HI_THRESHOLD
                    abs_max_iou = torch.argmax(ious, dim=0)
                    pos_mask[abs_max_iou] = 1
                    pos_mask = torch.nonzero(pos_mask)
                    neg_mask = torch.nonzero(max_iou < RPN_LO_THRESHOLD)
                    target_anchors = random_choice(neg_mask, batch_size)
                    rpn_pos_numel = int(batch_size * RPN_POS_RATIO)
                    if pos_mask.numel() < rpn_pos_numel:
                        rpn_pos_numel = pos_mask.numel()
                    target_anchors[:rpn_pos_numel] = random_choice(pos_mask, rpn_pos_numel)
                    target_cls = torch.zeros(batch_size, 2, device=self.device)
                    target_cls[:rpn_pos_numel, 0] = 1
                    target_cls[rpn_pos_numel:, 1] = 1
                    target_bboxes = target_bboxes[argmax_iou][target_anchors]
                    target_reg = parameterize_bboxes(target_bboxes, anchors[target_anchors])
                    cls = cls[target_anchors]
                    reg = parameterize_bboxes(reg[target_anchors], anchors[target_anchors])
                    loss = self.loss(reg, cls, target_reg, target_cls)
                    losses.append(loss['total'].item())
                if i > len(val_dataloader) * fraction:
                    break

        return np.mean(losses)


class FasterRCNN(nn.Module):

    def __init__(self,
                 anchors,
                 name='FasterR-CNN',
                 num_classes=20,
                 backbone=None,
                 device='cuda'):
        super(FasterRCNN, self).__init__()
        if backbone is None:
            backbone = VGGBackbone()
        self.fast_rcnn = FastRCNN(num_classes=num_classes,
                                  backbone=backbone,
                                  device=device)

        self.rpn = RPN(anchors=anchors,
                       device=device)

        self.name = name
        self.num_classes = num_classes

        self.iteration = 0

        self.device = device

    def forward(self, x):
        features = self.fast_rcnn.backbone.features(x)
        reg, cls = self.rpn(features)
        grid_size = torch.tensor(features.shape[-2:], device=self.device)
        rois, _ = self.rpn.post_process(reg, cls, grid_size, max_rois=MAX_TEST_RPN_ROIS, nms=True)
        reg, cls = self.fast_rcnn(features, rois, extract_features=False)
        bboxes, classes = self.fast_rcnn.post_process(reg, cls, rois)

        return bboxes, classes

    def predict(self, dataset, confidence_threshold=0.1, overlap_threshold=0.5, show=True, export=True):

        self.eval()

        dataloader = DataLoader(dataset=dataset,
                                shuffle=False,
                                num_workers=NUM_WORKERS)

        context = tqdm(total=len(dataloader), desc='Exporting', leave=True) if export else nullcontext()

        image_idx_ = []
        bboxes_ = []
        confidence_ = []
        classes_ = []

        with torch.no_grad():
            with context as pbar:
                for images, images_info in dataloader:
                    width = images_info['width'][0].to(self.device)
                    height = images_info['height'][0].to(self.device)
                    set_name = images_info['dataset'][0]
                    ids = images_info['id'][0]
                    images = images.to(self.device)
                    bboxes, cls = self(images)
                    (bboxes,
                     classes,
                     confidences) = self.fast_rcnn.process_bboxes(bboxes,
                                                                  cls,
                                                                  (width, height),
                                                                  confidence_threshold=confidence_threshold,
                                                                  overlap_threshold=overlap_threshold)

                    if show:
                        image = to_numpy_image(images[0], size=(width, height))
                        for bbox, cls, confidence in zip(bboxes, classes, confidences):
                            name = dataset.classes[cls]
                            add_bbox_to_image(image, bbox, confidence, name)
                        plt.imshow(image)
                        # plt.axis('off')
                        plt.show()

                    if export:
                        for bbox, cls, confidence in zip(bboxes, classes, confidences):
                            name = dataset.classes[cls]
                            confidence = confidence.item()
                            x1, y1, x2, y2 = bbox.cpu().numpy()
                            export_prediction(cls=name,
                                              prefix=self.name,
                                              image_id=ids,
                                              left=x1,
                                              top=y1,
                                              right=x2,
                                              bottom=y2,
                                              confidence=confidence,
                                              set_name=set_name)

                    bboxes_.append(bboxes)
                    confidence_.append(confidences)
                    classes_.append(classes)
                    image_idx_.append([ids] * len(bboxes))

                    pbar.update()

            if len(bboxes_) > 0:
                bboxes = torch.cat(bboxes_).view(-1, 4)
                classes = torch.cat(classes_).flatten()
                confidence = torch.cat(confidence_).flatten()
                image_idx = [item for sublist in image_idx_ for item in sublist]

                return bboxes, classes, confidence, image_idx
            else:
                return torch.tensor([], device=self.device), \
                       torch.tensor([], device=self.device), \
                       torch.tensor([], device=self.device), \
                       []

    def alternate_training(self, train_data, image_batch_size=2, roi_batch_size=64,
                           epochs=1, lr=1e-3, momentum=0.9, val_data=None, shuffle=True,
                           stage=None):

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
                                               lr=lr,
                                               image_batch_size=image_batch_size[i],
                                               roi_batch_size=roi_batch_size[i],
                                               shuffle=shuffle,
                                               stage=i)
                train_loss.append(loss[0])
                val_loss.append(loss[1])

            return train_loss, val_loss

        elif stage == 0:

            plist = [{'params': self.fast_rcnn.features.parameters()},
                     {'params': self.rpn.parameters()}
                     ]
            optimizer = optim.SGD(plist, lr=lr, momentum=momentum, weight_decay=1e-4)

            self.fast_rcnn.train()
            train_loss, val_loss = self.rpn.fit(train_data=train_data,
                                                backbone=self.fast_rcnn.features,
                                                optimizer=optimizer,
                                                scheduler=None,
                                                batch_size=roi_batch_size,
                                                epochs=epochs,
                                                val_data=val_data,
                                                shuffle=shuffle)

            self.save_model(self.name + '_{}.pkl'.format('stage_0'))

            return train_loss, val_loss

        elif stage == 1:

            backbone = copy.deepcopy(self.fast_rcnn.features)
            self.fast_rcnn = FastRCNN(num_classes=self.num_classes,
                                      pretrained=True,
                                      model_path=self.model_path,
                                      device=self.device)
            plist = [{'params': self.fast_rcnn.parameters()}]
            optimizer = optim.SGD(plist, lr=lr, momentum=momentum, weight_decay=1e-4)

            target_lr = lr
            initial_lr = 1e-4
            warm_up = 3
            step_size = 0.95
            step_frequency = 1
            gradient = (target_lr - initial_lr) / warm_up

            def f(e):
                if e < warm_up:
                    return gradient * e + initial_lr
                else:
                    return target_lr * step_size ** ((e - warm_up) // step_frequency)

            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)

            self.rpn.eval()
            train_loss, val_loss = self.fast_rcnn.fit(train_data=train_data,
                                                      optimizer=optimizer,
                                                      scheduler=scheduler,
                                                      alt_feature_extractor=backbone,
                                                      rpn=self.rpn,
                                                      roi_batch_size=roi_batch_size,
                                                      epochs=epochs,
                                                      val_data=val_data,
                                                      shuffle=shuffle)

            self.save_model(self.name + '_{}.pkl'.format('stage_1'))

            return train_loss, val_loss

        elif stage == 2:

            plist = [{'params': self.rpn.parameters()}]
            optimizer = optim.SGD(plist, lr=lr, momentum=momentum, weight_decay=1e-4)

            self.fast_rcnn.eval()
            self.fast_rcnn.freeze()
            train_loss, val_loss = self.rpn.fit(train_data=train_data,
                                                backbone=self.fast_rcnn.features,
                                                optimizer=optimizer,
                                                scheduler=None,
                                                batch_size=roi_batch_size,
                                                epochs=epochs,
                                                val_data=val_data,
                                                shuffle=shuffle)

            self.save_model(self.name + '_{}.pkl'.format('stage_2'))

            return train_loss, val_loss

        elif stage == 3:

            plist = [{'params': self.fast_rcnn.classifier.parameters()},
                     {'params': self.fast_rcnn.cls.parameters()},
                     {'params': self.fast_rcnn.reg.parameters()}]
            optimizer = optim.SGD(plist, lr=lr, momentum=momentum, weight_decay=5e-4)

            self.rpn.eval()
            self.fast_rcnn.freeze()
            train_loss, val_loss = self.fast_rcnn.fit(train_data=train_data,
                                                      optimizer=optimizer,
                                                      scheduler=None,
                                                      alt_feature_extractor=self.fast_rcnn.features,
                                                      rpn=self.rpn,
                                                      roi_batch_size=roi_batch_size,
                                                      epochs=epochs,
                                                      val_data=val_data,
                                                      shuffle=shuffle)

            self.save_model(self.name + '_{}.pkl'.format('stage_3'))

            return train_loss, val_loss

    def joint_training(self, train_data, optimizer, scheduler=None, image_batch_size=2,
                       rpn_batch_size=256, frcn_batch_size=64, epochs=1, val_data=None, shuffle=True, checkpoint_frequency=100):

        self.train()

        train_dataloader = DataLoader(dataset=train_data,
                                      shuffle=shuffle,
                                      num_workers=NUM_WORKERS)

        train_loss = []
        val_loss = []

        for epoch in range(1, epochs + 1):
            stats = {'avg_obj_iou': [], 'avg_class': [], 'avg_pobj': [], 'avg_pnoobj': []}
            rpn_loss = []
            frcn_loss = []
            with tqdm(total=len(train_dataloader),
                      desc='Epoch: [{}/{}]'.format(epoch, epochs),
                      leave=True,
                      unit='batches') as inner:
                for images, images_info, (images_bboxes, images_classes) in train_dataloader:
                    images = images.to(self.device)
                    images_bboxes = images_bboxes[0].to(self.device)
                    images_classes = images_classes[0].to(self.device)
                    if inner.n % image_batch_size == 0:
                        optimizer.zero_grad()
                    features = self.fast_rcnn.backbone.features(images)
                    if images_bboxes.numel() > 0:
                        reg, cls = self.rpn(features)
                        cls_og = cls.clone()
                        reg_og = reg.clone()
                        grid_size = torch.tensor(features.shape[-2:], device=self.device)
                        anchors = self.rpn.construct_anchors(grid_size)
                        valid = self.rpn.validate_anchors(anchors)  # This has the effect of scrambling anchors
                        cls = cls[0][valid]
                        reg = reg[0][valid]
                        anchors = anchors[valid]
                        ious = jaccard(anchors, images_bboxes)
                        max_iou, argmax_iou = torch.max(ious, dim=1)
                        pos_mask = max_iou > RPN_HI_THRESHOLD
                        abs_max_iou = torch.argmax(ious, dim=0)
                        pos_mask[abs_max_iou] = True
                        neg_mask = torch.nonzero((max_iou < RPN_LO_THRESHOLD) * (~pos_mask))
                        pos_mask = torch.nonzero(pos_mask)
                        samples, n_p, n_n = sample_ids(rpn_batch_size, pos_mask, neg_mask, RPN_POS_RATIO)
                        target_cls = torch.zeros(rpn_batch_size, 2, device=self.device)
                        target_cls[:n_p, 0] = 1.
                        target_cls[-n_n:, 1] = 1.
                        images_bboxes_ = images_bboxes[argmax_iou][samples[:n_p]]
                        target_reg = parameterize_bboxes(images_bboxes_, anchors[samples[:n_p]])
                        reg = reg[samples[:n_p]]
                        cls = cls[samples]
                        loss = self.rpn.loss(reg, cls, target_reg, target_cls, stats)
                        rpn_loss.append(loss['total'].item())
                        loss['total'].backward(retain_graph=True)
                        # FRCN
                        rois = self.rpn.post_process(reg_og, cls_og, grid_size, MAX_TRAIN_RPN_ROIS, True)[0]
                        ious = jaccard(rois, images_bboxes)
                        max_iou, argmax_iou = torch.max(ious, dim=1)
                        pos_mask = max_iou > FRCN_HI_THRESHOLD
                        abs_max_iou = torch.argmax(ious, dim=0)
                        pos_mask[abs_max_iou] = 1  # Added ROI with maximum IoU to pos samples.
                        pos_mask = torch.nonzero(pos_mask)
                        neg_mask = torch.nonzero((FRCN_LO_LO_THRESHOLD < max_iou) * (max_iou < FRCN_LO_HI_THRESHOLD))
                        samples, n_p, n_n = sample_ids(frcn_batch_size, pos_mask, neg_mask, FRCN_POS_RATIO)
                        frcn_target_rois = rois[samples]
                        frcn_target_classes = torch.zeros(frcn_batch_size, self.num_classes + 1, device=self.device)
                        frcn_target_classes[:n_p] = images_classes[argmax_iou[samples[:n_p]]]
                        frcn_target_classes[-n_n:, 0] = 1.
                        target_bboxes = parameterize_bboxes(images_bboxes[argmax_iou[samples[:n_p]]],
                                                            rois[samples[:n_p]])
                    reg, cls = self.fast_rcnn(features, frcn_target_rois, extract_features=False)
                    reg = reg[:n_p]
                    loss = self.fast_rcnn.loss(reg, cls, target_bboxes[None].detach(), frcn_target_classes[None].detach(), stats)
                    frcn_loss.append(loss['total'].item())
                    loss['total'].backward()

                    weights = np.arange(1, 1 + len(frcn_loss))
                    disp_str = ' RPN Training Loss: {:.6f}, '.format(np.average(rpn_loss, weights=weights)) + \
                               ' FRCN Training Loss: {:.6f}, '.format(np.average(frcn_loss, weights=weights)) + \
                               ' Avg IOU: {:.4f},  '.format(np.average(stats['avg_obj_iou'], weights=weights)) + \
                               ' Avg P|Obj: {:.4f},  '.format(np.average(stats['avg_pobj'], weights=weights)) + \
                               ' Avg P|Noobj: {:.4f},  '.format(np.average(stats['avg_pnoobj'], weights=weights)) + \
                               ' Avg Class: {:.4f}, '.format(np.average(stats['avg_class'], weights=weights)) + \
                               ' Iteration: {:d}'.format(self.iteration)
                    inner.set_postfix_str(disp_str)
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1., norm_type='inf')  # ???
                    if (inner.n + 1) % image_batch_size == 0:
                        optimizer.step()
                        if scheduler is not None:
                            scheduler.step()
                            self.iteration += 1
                    inner.update()
                train_loss.append([np.average(rpn_loss, weights=weights), np.average(frcn_loss, weights=weights)])
                if val_data is not None:
                    val_data.reset_image_size()
                    val_loss.append(self.calculate_loss(val_data, roi_batch_size))
                    inner.set_postfix_str(' RPN Training Loss: {:.6f},  '
                                          'FRCN Training Loss: {:.6f},  '
                                          'RPN Validation Loss: {:.6f},  '
                                          'FRCN Validation Loss: {:.6f}'.format(train_loss[-1][0],
                                                                                train_loss[-1][1],
                                                                                val_loss[-1][0],
                                                                                val_loss[-1][1]))
                else:
                    inner.set_postfix_str(' RPN Training Loss: {:.6f},  FRCN Training Loss: {:.6f}'.format(train_loss[-1][0], train_loss[-1][1]))
                with open('loss_joint.txt', 'a') as fl:
                    fl.writelines('Epoch {}: '
                                  'RPN Training Loss: {:.6f},  '
                                  'FRCN Training Loss: {:.6f}\n'.format(epoch,
                                                                        train_loss[-1][0],
                                                                        train_loss[-1][1]))
            if scheduler is not None:
                scheduler.step()
            if epoch % checkpoint_frequency == 0:
                self.save_model(self.name + '{}.pkl'.format(epoch))

        return train_loss, val_loss

    # def calculate_loss(self, data, roi_batch_size=128, fraction=0.01):
    #     """
    #     Calculates the loss for a random partition of a given dataset without
    #     tracking gradients. Useful for displaying the validation loss during
    #     training or the test loss during evaluation.
    #     Parameters
    #     ----------
    #     data : PascalDatasetImage
    #         A dataset object which returns images and image info to use for calculating
    #         the loss. Only a fraction of the images in the dataset will be tested.
    #     image_batch_size : int
    #         The number of images to load for each batch and for which the loss should be
    #         calculated. This should not influence the value that the function returns,
    #         but will affect performance.
    #     roi_batch_size : int
    #         The number of ROIs per image for which the loss should be calculated. This
    #         should not influence the value that the function returns, but will affect
    #         performance.
    #     fraction : float
    #         The fraction of images from data that the loss should be calculated for.
    #
    #     Returns
    #     -------
    #     tuple
    #         The mean RPN and FRCN loss over the fraction of the images that were sampled from
    #         the data.
    #     """
    #     val_dataloader = DataLoader(dataset=data,
    #                                 shuffle=True,
    #                                 num_workers=NUM_WORKERS)
    #     rpn_losses = []
    #     frcn_losses = []
    #
    #     with torch.no_grad():
    #         for i, (images, images_info, images_transforms) in enumerate(val_dataloader, 1):
    #             images = images.to(self.device)
    #             image_batch_size = len(images)
    #             frcn_target_rois = [torch.zeros(roi_batch_size, 4, device=self.device)] * image_batch_size
    #             frcn_target_classes = torch.zeros(image_batch_size, roi_batch_size, self.num_classes + 1,
    #                                               device=self.device)
    #             target_bboxes = torch.zeros(image_batch_size, roi_batch_size, 4, device=self.device)
    #             features = self.fast_rcnn.features(images)
    #             for i in range(image_batch_size):
    #                 image_info_copy = access_dict_list(images_info, i)
    #                 image_transforms_copy = access_dict_list(images_transforms, i)
    #                 image_bboxes, image_classes = data.get_gt_bboxes(image_info_copy, image_transforms_copy)
    #                 image_bboxes = image_bboxes.to(self.device)
    #                 image_classes = image_classes.to(self.device)
    #                 if image_bboxes.numel() > 0:
    #                     cls, reg = self.rpn(features[i][None])
    #                     cls_copy = cls.clone()
    #                     reg_copy = reg.clone()
    #                     grid_size = torch.tensor(features.shape[-2:], device=self.device)
    #                     anchors = self.rpn.construct_anchors(grid_size)
    #                     valid = self.rpn.validate_anchors(anchors)
    #                     cls = cls[0][valid]
    #                     reg = reg[0][valid]
    #                     anchors = anchors[valid]
    #                     reg = deparameterize_bboxes(reg, anchors)
    #                     ious = jaccard(anchors, image_bboxes)
    #                     max_iou, argmax_iou = torch.max(ious, dim=1)
    #                     pos_mask = max_iou > RPN_HI_THRESHOLD
    #                     abs_max_iou = torch.argmax(ious, dim=0)
    #                     pos_mask[abs_max_iou] = 1
    #                     pos_mask = torch.nonzero(pos_mask)
    #                     neg_mask = torch.nonzero(max_iou < RPN_LO_THRESHOLD)
    #                     target_anchors = random_choice(neg_mask, roi_batch_size)
    #                     rpn_pos_numel = int(roi_batch_size * RPN_POS_RATIO)
    #                     if pos_mask.numel() < rpn_pos_numel:
    #                         rpn_pos_numel = pos_mask.numel()
    #                     target_anchors[:rpn_pos_numel] = random_choice(pos_mask, rpn_pos_numel)
    #                     target_cls = torch.zeros(roi_batch_size, 2, device=self.device)
    #                     target_cls[:rpn_pos_numel, 0] = 1
    #                     target_cls[rpn_pos_numel:, 1] = 1
    #                     image_bboxes_copy = image_bboxes[argmax_iou][target_anchors]
    #                     target_reg = parameterize_bboxes(image_bboxes_copy, anchors[target_anchors])
    #                     cls = cls[target_anchors]
    #                     reg = parameterize_bboxes(reg[target_anchors], anchors[target_anchors])
    #                     loss = self.rpn.loss(cls, reg, target_cls, target_reg)
    #                     rpn_losses.append(loss['total'].item())
    #                     # FRCN
    #                     rois = self.rpn.post_process(cls_copy, reg_copy, grid_size, MAX_TRAIN_RPN_ROIS, True)[1][0]
    #                     ious = jaccard(rois, image_bboxes)
    #                     max_iou, argmax_iou = torch.max(ious, dim=1)
    #                     pos_mask = max_iou > FRCN_HI_THRESHOLD
    #                     abs_max_iou = torch.argmax(ious, dim=0)
    #                     pos_mask[abs_max_iou] = 1  # Added ROI with maximum IoU to pos samples.
    #                     pos_mask = torch.nonzero(pos_mask)
    #                     neg_mask = torch.nonzero(
    #                         (FRCN_LO_LO_THRESHOLD < max_iou) * (max_iou < FRCN_LO_HI_THRESHOLD))
    #                     if neg_mask.numel() > 0:
    #                         samples = random_choice(neg_mask, roi_batch_size, replace=True)
    #                     else:
    #                         neg_mask = torch.nonzero(max_iou < FRCN_LO_HI_THRESHOLD)
    #                         if neg_mask.numel() > 0:
    #                             samples = random_choice(neg_mask, roi_batch_size, replace=True)
    #                         else:
    #                             neg_mask = torch.nonzero(max_iou < FRCN_HI_THRESHOLD)
    #                             samples = random_choice(neg_mask, roi_batch_size, replace=True)
    #                     frcn_pos_numel = int(roi_batch_size * FRCN_POS_RATIO)
    #                     if pos_mask.numel() < frcn_pos_numel:
    #                         frcn_pos_numel = pos_mask.numel()
    #                     if frcn_pos_numel > 0:
    #                         samples[:frcn_pos_numel] = random_choice(pos_mask, frcn_pos_numel)
    #                     frcn_target_rois[i] = rois[samples].detach()
    #                     frcn_target_classes[i][:frcn_pos_numel] = image_classes[argmax_iou[samples[:frcn_pos_numel]]]
    #                     frcn_target_classes[i][frcn_pos_numel:, 0] = 1.
    #                     target_bboxes[i] = parameterize_bboxes(image_bboxes[argmax_iou[samples]], rois[samples])
    #             cls, reg = self.fast_rcnn(images, frcn_target_rois)
    #             loss = self.fast_rcnn.loss(cls, reg, frcn_target_classes.detach(), target_bboxes.detach())
    #             frcn_losses.append(loss['total'].item())
    #             if i > len(val_dataloader) * fraction:
    #                 break
    #
    #     return np.mean(rpn_losses), np.mean(frcn_losses)

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
