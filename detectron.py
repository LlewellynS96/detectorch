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
from utils import xyxy2xywh
from utils import to_numpy_image, add_bbox_to_image, nullcontext
from utils import NUM_WORKERS


NETWORK_STRIDE = 16
RPN_HI_THRESHOLD = 0.7
RPN_LO_THRESHOLD = 0.3
FRCN_HI_THRESHOLD = 0.5
FRCN_LO_LO_THRESHOLD = 0.1
FRCN_LO_HI_THRESHOLD = 0.5
RPN_POS_RATIO = 0.5
FRCN_POS_RATIO = 0.25
RPN_NMS_THRESHOLD = 0.7
PRE_NMS_TOPK_TRAIN = 12000
POST_NMS_TOPK_TRAIN = 2000
POST_NMS_TOPK_TEST = 300
PRE_NMS_TOPK_TEST = 6000
BBOX_REG_WEIGHTS = [10., 10., 5., 5.]


class ResNetBackbone(nn.Module):
    def __init__(self,
                 model=models.resnet50,
                 model_path='models/resnet50-19c8e357.pth',
                 pretrained=True,
                 input_dims=3,
                 device='cuda'):
        super(ResNetBackbone, self).__init__()
        self.device = device
        self.model = model()
        if pretrained:
            self.model_path = model_path
            state_dict = torch.load(self.model_path)
            self.model.load_state_dict({k: v for k, v in state_dict.items() if k in self.model.state_dict()})

        self.features = nn.Sequential(*[self.model.conv1,
                                        self.model.bn1,
                                        self.model.relu,
                                        self.model.maxpool,
                                        self.model.layer1,
                                        self.model.layer2,
                                        self.model.layer3])
        self.classifier = nn.Sequential(*[self.model.layer4,
                                          self.model.avgpool
                                          ])

        self.channels = 1024
        self.roi_pool_size = (7, 7)
        self.classifier_in_shape = (-1, self.channels, *self.roi_pool_size)
        self.out_features = 2048

        self.set_input_dims(input_dims)

        self.to(device)

    def mini_freeze(self):
        for layer in list(self.features)[:-1]:
            for param in layer.parameters():
                param.requires_grad = False

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                for param in module.parameters():
                    param.requires_grad = False
                module.eval()

    def set_input_dims(self, dims=3):
        modules = list(self.features)
        conv = modules[0]
        assert dims <= conv.weight.data.shape[1]
        modules[0] = nn.Conv2d(dims, 64, kernel_size=7, stride=2, padding=3, bias=False)
        modules[0].weight.data.copy_(conv.weight.data[:, :dims])
        self.features = nn.Sequential(*modules)


class VGGBackbone(nn.Module):
    def __init__(self,
                 model=models.vgg16,
                 model_path='models/vgg16-397923af.pth',
                 pretrained=True,
                 use_dropout=False,
                 input_dims=3,
                 device='cuda'):
        super(VGGBackbone, self).__init__()
        self.device = device
        self.model = model()
        if pretrained:
            self.model_path = model_path
            state_dict = torch.load(self.model_path)
            self.model.load_state_dict({k: v for k, v in state_dict.items() if k in self.model.state_dict()})
        self.model.features = nn.Sequential(*list(self.model.features.children())[:-1])
        classifier = list(self.model.classifier.children())[:-2]
        if not use_dropout:
            del classifier[2]
        self.model.classifier = nn.Sequential(*classifier)

        self.features = self.model.features
        self.classifier = self.model.classifier

        self.channels = 512
        self.roi_pool_size = (7, 7)
        self.classifier_in_shape = (1, -1, self.channels * np.product(self.roi_pool_size))
        self.out_features = 4096

        self.set_input_dims(input_dims)

        self.to(device)

    def mini_freeze(self):
        """
            Freezes the first few layers of the backbone (conv and bn) of the VGG network.
        """
        for param in list(self.features.parameters())[:2*4]:
            param.requires_grad = False

    def set_input_dims(self, dims=3):
        modules = list(self.features)
        conv = modules[0]
        assert dims <= conv.weight.data.shape[1]
        modules[0] = nn.Conv2d(dims, 64, kernel_size=3)
        modules[0].weight.data.copy_(conv.weight.data[:, :dims])
        modules[0].bias.data.copy_(conv.bias.data)
        self.features = nn.Sequential(*modules)


class FastRCNN(nn.Module):

    def __init__(self,
                 num_classes=20,
                 backbone=None,
                 use_global_ctx=False,
                 device='cuda'):
        super(FastRCNN, self).__init__()
        self.global_context = use_global_ctx
        if backbone is None:
            backbone = VGGBackbone()
        self.num_classes = num_classes
        self.device = device
        self.backbone = backbone
        num_features = 2 * backbone.out_features if self.global_context else backbone.out_features
        self.reg = nn.Linear(num_features, self.num_classes * 4)
        self.reg.weight.data.normal_(0., 0.01)
        self.reg.bias.data.fill_(0)

        self.cls = nn.Linear(num_features, self.num_classes + 1)
        self.cls.weight.data.normal_(0., 0.01)
        self.cls.bias.data.fill_(0)

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
        # roi_features = ops.roi_pool(features, [rois_idxs], output_size=self.backbone.roi_pool_size)
        roi_features = ops.roi_align(features, [rois_idxs], output_size=self.backbone.roi_pool_size)
        roi_features = roi_features.view(*self.backbone.classifier_in_shape)
        roi_features = self.backbone.classifier(roi_features)
        roi_features = roi_features.reshape(1, -1, self.backbone.out_features)
        if self.global_context:
            tensor = torch.tensor([[0., 0., grid_size[1], grid_size[0]]], device=self.device)
            global_context = ops.roi_pool(features, [tensor], output_size=self.backbone.roi_pool_size)
            # global_context = ops.roi_align(features, [tensor], output_size=self.backbone.roi_pool_size)
            global_context = global_context.view(*self.backbone.classifier_in_shape)
            global_context = self.backbone.classifier(global_context)
            global_context = global_context.repeat_interleave(len(rois_idxs), dim=0)
            global_context = global_context.reshape(1, -1, self.backbone.out_features)
            roi_features = torch.cat((roi_features, global_context), dim=-1)
        reg = self.reg(roi_features)
        cls = self.cls(roi_features)

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

    def fit(self,
            train_data,
            optimizer,
            scheduler=None,
            alt_feature_extractor=None,
            rpn=None,
            rois=None,
            image_batch_size=2,
            frcn_batch_size=64,
            epochs=1,
            val_data=None,
            shuffle=True,
            checkpoint_frequency=100):
        assert rpn or rois
        self.train()
        if hasattr(self.fast_rcnn.backbone, 'freeze_bn'):
            self.fast_rcnn.backbone.freeze_bn()
        if alt_feature_extractor is None and hasattr(alt_feature_extractor, 'freeze_bn'):
            alt_feature_extractor.freeze_bn()

        train_dataloader = DataLoader(dataset=train_data,
                                      shuffle=shuffle,
                                      num_workers=NUM_WORKERS)

        train_losses = []
        val_losses = []

        for epoch in range(1, epochs + 1):
            stats = {'avg_frcn_class': [],
                     'avg_frcn_background': []}
            frcn_losses = []
            with tqdm(total=len(train_dataloader),
                      desc='Epoch: [{}/{}]'.format(epoch, epochs),
                      leave=True,
                      unit='batches') as inner:
                for images, _, (images_bboxes, images_classes) in train_dataloader:
                    images = images.to(self.device)
                    images_bboxes = images_bboxes[0].to(self.device)
                    images_classes = images_classes[0].to(self.device)
                    if inner.n % image_batch_size == 0:
                        optimizer.zero_grad()
                    if alt_feature_extractor is None:
                        features = self.backbone.features(images)
                    else:
                        with torch.no_grad():
                            features = alt_feature_extractor(images)
                    if rpn is not None:
                        reg, cls = rpn(features)
                        image_size = images.shape[-2:]
                        grid_size = features.shape[-2:]
                        anchors_xyxy = rpn.construct_anchors(image_size, grid_size)
                        valid = rpn.validate_anchors(anchors_xyxy)
                        reg = reg[:, valid]
                        cls = cls[:, valid]
                        anchors_xyxy = anchors_xyxy[valid]
                        anchors_xywh = xyxy2xywh(anchors_xyxy)
                        rois = self.rpn.post_process(reg,
                                                     cls,
                                                     anchors_xywh,
                                                     PRE_NMS_TOPK_TRAIN,
                                                     POST_NMS_TOPK_TRAIN,
                                                     True)[0]
                    # FRCN
                    samples, roi_match = self.sample(frcn_batch_size, images_bboxes, rois, stats)
                    rois_xyxy = rois[samples]
                    reg, cls = self(features, rois_xyxy.detach(), extract_features=False)
                    images_bboxes = images_bboxes[roi_match]
                    images_classes = images_classes[roi_match]
                    rois_xywh = xyxy2xywh(rois_xyxy)
                    loss, stats = self.fast_rcnn.loss(reg, cls, images_bboxes, images_classes, rois_xywh.detach(),
                                                      stats)
                    frcn_loss = loss['total']
                    frcn_losses.append(frcn_loss.item())
                    frcn_loss.backward()
                    weights = np.arange(1, 1 + len(frcn_losses))
                    disp_str = ' Training Loss: {:.5f}, '.format(np.average(frcn_losses, weights=weights)) + \
                               ' Avg Class: {:.3f},  '.format(np.average(stats['avg_frcn_class'], weights=weights)) + \
                               ' Avg P|Noobj: {:.3f}'.format(np.average(stats['avg_frcn_background'], weights=weights))
                    inner.set_postfix_str(disp_str)
                    if (inner.n + 1) % image_batch_size == 0:
                        optimizer.step()
                        if scheduler is not None:
                            scheduler.step()
                        self.iteration += 1
                    inner.update()
                train_losses.append(np.average(frcn_losses, weights=weights))
                if val_data is not None:
                    # val_loss.append(self.calculate_loss(val_data, roi_batch_size))
                    pass
                else:
                    disp_str = ' Training Loss: {:.5f}, '.format(np.average(frcn_losses, weights=weights)) + \
                               ' Avg Class: {:.3f},  '.format(np.average(stats['avg_frcn_class'], weights=weights)) + \
                               ' Avg P|Noobj: {:.3f}'.format(np.average(stats['avg_frcn_background'], weights=weights))
                    inner.set_postfix_str(disp_str)
                with open('training_loss.txt', 'a') as fl:
                    fl.writelines('Epoch: {} '.format(epoch) + disp_str + '\n')
            if epoch % checkpoint_frequency == 0:
                self.save_model(self.name + '{}.pkl'.format(epoch))

        return train_losses, val_losses

    @staticmethod
    def sample(n, images_bboxes, rois, stats):
        if len(images_bboxes) > 0:
            ious = jaccard(rois, images_bboxes)
            max_iou, argmax_iou = torch.max(ious, dim=1)
            pos_mask = max_iou > FRCN_HI_THRESHOLD
            abs_max_iou, abs_argmax_iou = torch.max(ious, dim=0)
            stats['avg_rpn_iou'].append(abs_max_iou.mean().item())
            pos_mask[abs_argmax_iou] = True  # Added ROI with maximum IoU to pos samples.
            neg_mask = torch.nonzero((FRCN_LO_LO_THRESHOLD <= max_iou) * (max_iou < FRCN_LO_HI_THRESHOLD) * (~pos_mask))
            pos_mask = torch.nonzero(pos_mask)
            if len(neg_mask) < n:
                neg_mask = torch.nonzero(max_iou < FRCN_LO_HI_THRESHOLD)
            samples, n_p, _ = sample_ids(n, pos_mask, neg_mask, FRCN_POS_RATIO)
            roi_match = argmax_iou[samples[:n_p]]
        else:
            stats['avg_rpn_iou'].append(0.)
            samples = torch.randperm(len(rois))[:n]
            roi_match = []

        return samples, roi_match

    def loss(self, reg, cls, target_bboxes, target_classes, rois, stats):
        assert reg.shape[0] == 1
        assert cls.shape[0] == 1

        loss = dict()

        reg = reg[0]
        cls = cls[0]

        batch_size = len(reg)
        num_obj = len(target_bboxes)

        if num_obj > 0:
            target_reg = parameterize_bboxes(target_bboxes,
                                             rois[:num_obj])

        target_cls = torch.zeros(batch_size, self.num_classes + 1, device=self.device)
        if num_obj > 0:
            target_cls[:num_obj] = target_classes
        target_cls[num_obj:, 0] = 1.

        target_cls = torch.argmax(target_cls, dim=-1)
        obj_mask = torch.nonzero(target_cls > 0)[:, 0]

        lambd = 1. / cls.shape[0]

        cls = F.log_softmax(cls, dim=-1)
        loss['cls'] = lambd * nn.NLLLoss(reduction='sum')(cls, target_cls)

        arange = torch.arange(num_obj)
        if num_obj > 0:
            stats['avg_frcn_class'].append(torch.exp(cls[:num_obj][arange, target_cls[:num_obj]]).mean().item())
        else:
            stats['avg_frcn_class'].append(0.)
        stats['avg_frcn_background'].append(torch.exp(cls[num_obj:, 0]).mean().item())

        if num_obj > 0:
            cls = (target_cls - 1)[obj_mask]
            bbox_reg_weights = torch.tensor(BBOX_REG_WEIGHTS, device=self.device)
            loss['reg'] = lambd * nn.SmoothL1Loss(reduction='sum')(reg[obj_mask, cls] * bbox_reg_weights,
                                                                   target_reg[obj_mask] * bbox_reg_weights)
        else:
            loss['reg'] = 0.

        loss['total'] = loss['cls'] + loss['reg']

        return loss, stats

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
                   torch.tensor([], dtype=torch.int64, device=self.device), \
                   torch.tensor([], device=self.device)

        confidence, classes = torch.max(classes, dim=-1)

        mask = confidence > confidence_threshold

        if sum(mask) == 0:
            return torch.tensor([], device=self.device), \
                   torch.tensor([], dtype=torch.int64, device=self.device), \
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

            # mask = torchvision.ops.nms(bboxes, confidence, overlap_threshold)
            # bboxes_tmp.append(bboxes[mask])
            # classes_tmp.append(classes[mask])
            # confidence_tmp.append(confidence[mask])

        if len(bboxes) > 0:
            bboxes = torch.cat(bboxes_tmp).view(-1, 4)
            bboxes[:, ::2] *= image_size[0]
            bboxes[:, 1::2] *= image_size[1]
            classes = torch.cat(classes_tmp).flatten()
            confidence = torch.cat(confidence_tmp).flatten()

            return bboxes, classes, confidence
        else:
            return torch.tensor([], device=self.device), \
                   torch.tensor([], dtype=torch.int64, device=self.device), \
                   torch.tensor([], device=self.device)

    def mini_freeze(self):
        """
            Freezes the first few layers of the backbone (conv and bn) of the VGG network.
        """
        self.backbone.mini_freeze()


class RPN(nn.Module):

    def __init__(self, anchors, in_channels, device='cuda'):

        super(RPN, self).__init__()
        self.in_channels = in_channels
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

        for layer in [self.sliding_windows, self.cls, self.reg]:
            layer.weight.data.normal_(0., 0.01)
            layer.bias.data.fill_(0)

        self.iteration = 0

        self.to(device)

    def forward(self, x):
        assert x.shape[0] == 1

        features = F.relu(self.sliding_windows(x))
        reg = self.reg(features)
        cls = self.cls(features)

        reg = reg.permute(0, 2, 3, 1).reshape(1, -1, 4)
        cls = cls.permute(0, 2, 3, 1).reshape(1, -1, 2)

        return reg, cls

    @staticmethod
    def post_process(reg, cls, anchors_xywh, pre_nms_topk=12000, post_nms_topk=2000, nms_threshold=0.7):
        assert reg.shape[0] == 1
        assert cls.shape[0] == 1

        reg = reg[0]
        cls = cls[0]

        classes = F.softmax(cls, dim=-1)
        num_bboxes = len(anchors_xywh)
        if num_bboxes > pre_nms_topk:
            # sort is faster than topk (https://github.com/pytorch/pytorch/issues/22812)
            # keep = torch.topk(classes[:, 0], pre_nms_topk)[1]
            keep = torch.argsort(classes[:, 0])[-pre_nms_topk:]
            bboxes = deparameterize_bboxes(reg[keep], anchors_xywh[keep])
            classes = classes[keep]
        else:
            bboxes = deparameterize_bboxes(reg, anchors_xywh)

        bboxes = torch.clamp(bboxes, min=0., max=1.)

        keep = torchvision.ops.nms(bboxes, classes[:, 0], nms_threshold)
        classes = classes[keep]
        bboxes = bboxes[keep]
        classes = classes[:post_nms_topk]
        bboxes = bboxes[:post_nms_topk]

        return bboxes, classes

    def construct_anchors(self, image_size, grid_size):

        anchors = torch.zeros((*grid_size, self.num_anchors, 4), device=self.device)

        x = torch.arange(0, grid_size[1], device=self.device)
        y = torch.arange(0, grid_size[0], device=self.device)

        xx, yy = torch.meshgrid(x, y)

        x_coords = x.reshape(1, -1, 1, 1).to(torch.float32)
        y_coords = y.reshape(-1, 1, 1, 1).to(torch.float32)

        anchors[yy, xx, :, :2] = -self.anchors / 2.
        anchors[yy, xx, :, 2:] = self.anchors / 2.

        anchors[:, x, :, ::2] += x_coords
        anchors[:, x, :, ::2] *= (NETWORK_STRIDE / image_size[1])
        anchors[y, :, :, 1::2] += y_coords
        anchors[y, :, :, 1::2] *= (NETWORK_STRIDE / image_size[0])

        anchors = anchors.reshape(-1, 4)

        return anchors

    @staticmethod
    def validate_anchors(anchors):
        keep = torch.nonzero((anchors[:, 0] >= 0) & (anchors[:, 1] >= 0) & (anchors[:, 2] <= 1) & (anchors[:, 3] <= 1))
        return keep.squeeze()

    def fit(self, train_data, backbone, optimizer, scheduler=None, image_batch_size=2, batch_size=64, epochs=1,
            val_data=None, shuffle=True):

        self.train()

        train_dataloader = DataLoader(dataset=train_data,
                                      batch_size=1,
                                      shuffle=shuffle,
                                      num_workers=NUM_WORKERS)

        train_loss = []
        val_loss = []

        for epoch in range(1, epochs + 1):
            rpn_loss = []
            stats = {'avg_rpn_iou': [],
                     'avg_anch_iou': [],
                     'avg_rpn_obj': [],
                     'avg_rpn_noobj': []}
            with tqdm(total=len(train_dataloader),
                      desc='Epoch: [{}/{}]'.format(epoch, epochs),
                      leave=True,
                      unit='batches') as inner:
                for images, _, (images_bboxes, _) in train_dataloader:
                    images = images.to(self.device)
                    images_bboxes = images_bboxes[0].to(self.device)
                    if inner.n % image_batch_size == 0:
                        optimizer.zero_grad()
                    features = backbone(images)
                    reg, cls = self(features)
                    grid_size = features.shape[-2:]
                    anchors_xyxy = self.construct_anchors(grid_size)
                    valid = self.validate_anchors(anchors_xyxy)  # This has the effect of scrambling anchors
                    reg = reg[:, valid]
                    cls = cls[:, valid]
                    anchors_xyxy = anchors_xyxy[valid]
                    anchors_xywh = xyxy2xywh(anchors_xyxy)
                    samples, roi_match = self.sample(batch_size, images_bboxes, anchors_xyxy, stats)
                    loss, stats = self.loss(reg[:, samples],
                                            cls[:, samples],
                                            images_bboxes[roi_match],
                                            anchors_xywh[samples].detach(),
                                            stats)
                    loss['total'].backward()
                    weights = np.arange(1, 1 + len(rpn_loss))
                    disp_str = ' RPN Training Loss (R/F): {:.5f}, '.format(np.average(rpn_loss, weights=weights)) + \
                               ' Avg IOU: {:.3f},  '.format(np.average(stats['avg_anch_iou'], weights=weights)) + \
                               ' Avg P|Obj: {:.3f},  '.format(np.average(stats['avg_rpn_obj'], weights=weights)) + \
                               ' Avg P|Noobj: {:.3f}'.format(np.average(stats['avg_rpn_noobj'], weights=weights))
                    inner.set_postfix_str(disp_str)
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1., norm_type='inf')  # ???
                    if (inner.n + 1) % image_batch_size == 0:
                        optimizer.step()
                        if scheduler is not None:
                            scheduler.step()
                        self.iteration += 1
                    inner.update()
                train_loss.append(np.average(rpn_loss, weights=weights))
                if val_data is not None:
                    pass
            with open('loss_rpn.txt', 'a') as fl:
                fl.writelines('Epoch {}: RPN Training Loss: {:.5f}\n'.format(epoch,
                                                                             train_loss[-1]))

        return train_loss, val_loss

    @staticmethod
    def sample(n, target_bboxes, anchors_xyxy, stats):
        if len(target_bboxes) > 0:
            ious = jaccard(anchors_xyxy, target_bboxes)
            max_iou, argmax_iou = torch.max(ious, dim=1)
            pos_mask = max_iou > RPN_HI_THRESHOLD
            abs_max_iou, abs_argmax_iou = torch.max(ious, dim=0)
            stats['avg_anch_iou'].append(abs_max_iou.mean().item())
            pos_mask[abs_argmax_iou] = True
            neg_mask = torch.nonzero((max_iou < RPN_LO_THRESHOLD) * (~pos_mask))
            pos_mask = torch.nonzero(pos_mask)
            samples, n_p, _ = sample_ids(n, pos_mask, neg_mask, RPN_POS_RATIO)
            anchor_match = argmax_iou[samples[:n_p]]
        else:
            stats['avg_anch_iou'].append(0.)
            samples = torch.randperm(len(anchors_xyxy))[:n]
            anchor_match = []

        return samples, anchor_match

    def loss(self, reg, cls, target_bboxes, anchors_xywh, stats, beta=1/9.):
        loss = dict()

        cls = cls[0].clone()
        reg = reg[0].clone()

        batch_size = len(reg)
        num_obj = len(target_bboxes)

        target_cls = torch.zeros(batch_size, 2, device=self.device)
        target_cls[:num_obj, 0] = 1.
        target_cls[num_obj:, 1] = 1.

        lambd = 1. / cls.shape[0]

        target_cls = torch.argmax(target_cls, dim=-1)
        cls = F.log_softmax(cls, dim=-1)
        loss['cls'] = lambd * nn.NLLLoss(reduction='sum')(cls, target_cls)
        if num_obj > 0:
            stats['avg_rpn_obj'].append(torch.exp(cls[:num_obj, 0]).mean().item())
        else:
            stats['avg_rpn_obj'].append(0.)
        stats['avg_rpn_noobj'].append(torch.exp(cls[num_obj:, 1]).mean().item())

        if num_obj > 0:
            obj_mask = torch.nonzero(target_cls == 0)[:, 0]

            target_reg = parameterize_bboxes(target_bboxes, anchors_xywh[obj_mask])

            loss['reg'] = lambd * beta * nn.SmoothL1Loss(reduction='sum')(reg[obj_mask] / beta,
                                                                          target_reg / beta)
        else:
            loss['reg'] = 0.

        loss['total'] = loss['cls'] + loss['reg']

        return loss, stats

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

    def predict(self, backbone, dataset, max_rois=10, overlap_threshold=0.7, show=True, export=False):

        self.eval()

        dataloader = DataLoader(dataset=dataset,
                                shuffle=False,
                                num_workers=NUM_WORKERS)

        context = tqdm(total=len(dataloader), desc='Exporting', leave=True) if export else nullcontext()

        image_idx_ = []
        rois_ = []
        confidence_ = []

        with torch.no_grad():
            with context as pbar:
                for images, images_info in dataloader:
                    width = images_info['width'][0].to(self.device)
                    height = images_info['height'][0].to(self.device)
                    set_name = images_info['dataset'][0]
                    ids = images_info['id'][0]
                    images = images.to(self.device)
                    features = backbone.features(images)
                    reg, cls = self(features)
                    image_size = images.shape[-2:]
                    grid_size = features.shape[-2:]
                    anchors_xyxy = self.construct_anchors(image_size, grid_size)
                    valid = self.validate_anchors(anchors_xyxy)  # This has the effect of scrambling anchors
                    reg = reg[:, valid]
                    cls = cls[:, valid]
                    anchors_xyxy = anchors_xyxy[valid]
                    anchors_xywh = xyxy2xywh(anchors_xyxy)
                    rois, classes = self.post_process(reg,
                                                      cls,
                                                      anchors_xywh,
                                                      PRE_NMS_TOPK_TEST,
                                                      max_rois,
                                                      nms_threshold=overlap_threshold)
                    rois[:, ::2] *= width
                    rois[:, 1::2] *= height
                    confidences = classes[:, 0]

                    if show:
                        image = to_numpy_image(images[0], size=(width, height))
                        for i, (roi, confidence) in reversed(list(enumerate(zip(rois, confidences), 1))):
                            add_bbox_to_image(image, roi, confidence, str(i))
                        plt.imshow(image)
                        # plt.axis('off')
                        plt.show()

                    if export:
                        for roi, confidence in zip(rois, confidences):
                            confidence = confidence.item()
                            x1, y1, x2, y2 = roi.cpu().numpy()
                            export_prediction(cls='roi',
                                              prefix=self.name,
                                              image_id=ids,
                                              left=x1,
                                              top=y1,
                                              right=x2,
                                              bottom=y2,
                                              confidence=confidence,
                                              set_name=set_name)

                    rois_.append(rois)
                    confidence_.append(confidences)
                    image_idx_.append([ids] * len(rois))

                    if pbar is not None:
                        pbar.update()

            if len(rois_) > 0:
                rois = torch.cat(rois_).view(-1, 4)
                confidence = torch.cat(confidence_).flatten()
                image_idx = [item for sublist in image_idx_ for item in sublist]

                return rois, confidence, image_idx
            else:
                return torch.tensor([], device=self.device), \
                       torch.tensor([], device=self.device), \
                       torch.tensor([], device=self.device), \
                       []


class FasterRCNN(nn.Module):

    def __init__(self,
                 anchors,
                 name='FasterR-CNN',
                 num_classes=20,
                 backbone=None,
                 use_global_ctx=False,
                 device='cuda'):
        super(FasterRCNN, self).__init__()
        if backbone is None:
            backbone = VGGBackbone()
        self.fast_rcnn = FastRCNN(num_classes=num_classes,
                                  backbone=backbone,
                                  use_global_ctx=use_global_ctx,
                                  device=device)

        self.rpn = RPN(anchors=anchors,
                       in_channels=self.fast_rcnn.backbone.channels,
                       device=device)

        self.name = name
        self.num_classes = num_classes

        self.iteration = 0

        self.device = device

    def forward(self, x):
        features = self.fast_rcnn.backbone.features(x)
        reg, cls = self.rpn(features)
        image_size = x.shape[-2:]
        grid_size = features.shape[-2:]
        anchors_xyxy = self.rpn.construct_anchors(image_size, grid_size)
        valid = self.rpn.validate_anchors(anchors_xyxy)  # This has the effect of scrambling anchors
        reg = reg[:, valid]
        cls = cls[:, valid]
        anchors_xyxy = anchors_xyxy[valid]
        anchors_xywh = xyxy2xywh(anchors_xyxy)
        rois, _ = self.rpn.post_process(reg,
                                        cls,
                                        anchors_xywh,
                                        PRE_NMS_TOPK_TEST,
                                        POST_NMS_TOPK_TEST,
                                        nms_threshold=RPN_NMS_THRESHOLD)
        reg, cls = self.fast_rcnn(features, rois, extract_features=False)
        rois = xyxy2xywh(rois)
        bboxes, classes = self.fast_rcnn.post_process(reg, cls, rois)

        return bboxes, classes

    def predict(self, dataset, confidence_threshold=0.1, overlap_threshold=0.5, show=True, export=True, gt=False):

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
                for data in dataloader:
                    if gt:
                        images, images_info, (images_bboxes, images_classes) = data
                        images_bboxes = images_bboxes.to(self.device)
                    else:
                        images, images_info = data
                    width = images_info['width'][0].to(self.device)
                    height = images_info['height'][0].to(self.device)
                    set_name = images_info['dataset'][0]
                    ids = images_info['id'][0]
                    images = images.to(self.device)
                    bboxes, cls = self(images)
                    # bboxes = images_bboxes[0].to(self.device)
                    # cls = images_classes[0].to(self.device)
                    (bboxes,
                     classes,
                     confidences) = self.fast_rcnn.process_bboxes(bboxes,
                                                                  cls,
                                                                  (width, height),
                                                                  confidence_threshold=confidence_threshold,
                                                                  overlap_threshold=overlap_threshold)

                    if show:
                        if images[0].shape[0] == 3:
                            image = to_numpy_image(images[0], size=(width, height), mu=dataset.mu, sigma=dataset.sigma)
                        else:
                            mu = dataset.mu[0]
                            sigma = dataset.sigma[0]
                            image = to_numpy_image(images[0][0], size=(width, height), mu=mu, sigma=sigma, normalised=False)
                        for bbox, cls, confidence in zip(bboxes, classes, confidences):
                            name = dataset.classes[cls]
                            if gt:
                                ious = jaccard(bbox[None], images_bboxes[0])
                                max_iou, _ = torch.max(ious, dim=1)
                                if max_iou >= 0.5:
                                    add_bbox_to_image(image, bbox, confidence, name, 2, [0., 255., 0.])
                                else:
                                    add_bbox_to_image(image, bbox, confidence, name, 2, [0., 255., 0.])
                            else:
                                add_bbox_to_image(image, bbox, confidence, name)
                        plt.imshow(image)
                        plt.axis('off')
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

                    if pbar is not None:
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

    def joint_training(self,
                       train_data,
                       optimizer,
                       scheduler=None,
                       image_batch_size=2,
                       rpn_batch_size=256,
                       frcn_batch_size=64,
                       epochs=1,
                       val_data=None,
                       shuffle=True,
                       checkpoint_frequency=100):

        self.train()
        if hasattr(self.fast_rcnn.backbone, 'freeze_bn'):
            self.fast_rcnn.backbone.freeze_bn()

        train_dataloader = DataLoader(dataset=train_data,
                                      shuffle=shuffle,
                                      num_workers=NUM_WORKERS)

        all_train_losses = []
        all_train_stats = []
        all_val_losses = []
        all_val_stats = []

        if val_data is not None:
            self.eval()
            val_losses, val_stats = self.calculate_loss(val_data, rpn_batch_size, frcn_batch_size, fraction=0.5)
            all_val_losses.append(val_losses)
            all_val_stats.append(val_stats)
            self.train()
            if hasattr(self.fast_rcnn.backbone, 'freeze_bn'):
                self.fast_rcnn.backbone.freeze_bn()

        for epoch in range(1, epochs + 1):
            batch_stats = {'avg_rpn_iou': [],
                           'avg_anch_iou': [],
                           'avg_rpn_obj': [],
                           'avg_frcn_class': [],
                           'avg_rpn_noobj': [],
                           'avg_frcn_background': []}
            val_stats = {'avg_rpn_iou': [],
                         'avg_anch_iou': [],
                         'avg_rpn_obj': [],
                         'avg_frcn_class': [],
                         'avg_rpn_noobj': [],
                         'avg_frcn_background': []}
            rpn_losses = []
            frcn_losses = []
            with tqdm(total=len(train_dataloader),
                      desc='Epoch: [{}/{}]'.format(epoch, epochs),
                      leave=True,
                      unit='batches') as inner:
                for images, _, (images_bboxes, images_classes) in train_dataloader:
                    images = images.to(self.device)
                    images_bboxes = images_bboxes[0].to(self.device)
                    images_classes = images_classes[0].to(self.device)
                    if inner.n % image_batch_size == 0:
                        optimizer.zero_grad()
                    features = self.fast_rcnn.backbone.features(images)
                    reg, cls = self.rpn(features)
                    image_size = images.shape[-2:]
                    grid_size = features.shape[-2:]
                    anchors_xyxy = self.rpn.construct_anchors(image_size, grid_size)
                    valid = self.rpn.validate_anchors(anchors_xyxy)  # This has the effect of scrambling anchors
                    reg = reg[:, valid]
                    cls = cls[:, valid]
                    anchors_xyxy = anchors_xyxy[valid]
                    anchors_xywh = xyxy2xywh(anchors_xyxy)
                    samples, roi_match = self.rpn.sample(rpn_batch_size, images_bboxes, anchors_xyxy, batch_stats)
                    loss, batch_stats = self.rpn.loss(reg[:, samples],
                                                      cls[:, samples],
                                                      images_bboxes[roi_match],
                                                      anchors_xywh[samples].detach(),
                                                      batch_stats)
                    rpn_loss = loss['total']
                    rois = self.rpn.post_process(reg, cls, anchors_xywh, PRE_NMS_TOPK_TRAIN, POST_NMS_TOPK_TRAIN, True)[0]
                    # FRCN
                    samples, roi_match = self.fast_rcnn.sample(frcn_batch_size, images_bboxes, rois, batch_stats)
                    rois_xyxy = rois[samples]
                    reg, cls = self.fast_rcnn(features, rois_xyxy.detach(), extract_features=False)
                    images_bboxes = images_bboxes[roi_match]
                    images_classes = images_classes[roi_match]
                    rois_xywh = xyxy2xywh(rois_xyxy)
                    loss, batch_stats = self.fast_rcnn.loss(reg, cls, images_bboxes, images_classes, rois_xywh.detach(), batch_stats)
                    frcn_loss = loss['total']
                    rpn_losses.append(rpn_loss.item())
                    frcn_losses.append(frcn_loss.item())
                    (rpn_loss + frcn_loss).backward()
                    weights = np.arange(1, 1 + len(frcn_losses))
                    disp_str = ' Training Loss (R/F): ' \
                               '{:.5f}/{:.5f}, '.format(np.average(rpn_losses, weights=weights),
                                                        np.average(frcn_losses, weights=weights)) + \
                               ' Avg IOU (R): {:.3f},  '.format(np.average(np.average(batch_stats['avg_rpn_iou'], weights=weights))) + \
                               ' Avg Class (R/F): ' \
                               '{:.3f}/{:.3f},  '.format(np.average(batch_stats['avg_rpn_obj'], weights=weights),
                                                         np.average(batch_stats['avg_frcn_class'], weights=weights)) + \
                               ' Avg P|Noobj (R/F): ' \
                               '{:.3f}/{:.3f}'.format(np.average(batch_stats['avg_rpn_noobj'], weights=weights),
                                                      np.average(batch_stats['avg_frcn_background'], weights=weights))
                    inner.set_postfix_str(disp_str)
                    if (inner.n + 1) % image_batch_size == 0:
                        # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10., norm_type='inf')  # ???
                        optimizer.step()
                        if scheduler is not None:
                            scheduler.step()
                        self.iteration += 1
                    inner.update()
                all_train_losses.append([rpn_losses, frcn_losses])
                all_train_stats.append(batch_stats)
                if val_data is not None:
                    self.eval()
                    val_losses, val_stats = self.calculate_loss(val_data, rpn_batch_size, frcn_batch_size, val_stats, fraction=0.5)
                    all_val_losses.append(val_losses)
                    all_val_stats.append(val_stats)
                    self.train()
                    if hasattr(self.fast_rcnn.backbone, 'freeze_bn'):
                        self.fast_rcnn.backbone.freeze_bn()
                    disp_str = ' Training Loss (R/F): ' \
                               '{:.5f}/{:.5f}, '.format(np.average(rpn_losses, weights=weights),
                                                        np.average(frcn_losses, weights=weights)) + \
                               ' Avg IOU (R): {:.3f},  '.format(
                                   np.average(batch_stats['avg_rpn_iou'], weights=weights)) + \
                               ' Avg Class (R/F): ' \
                               '{:.3f}/{:.3f},  '.format(np.average(batch_stats['avg_rpn_obj'], weights=weights),
                                                         np.average(batch_stats['avg_frcn_class'], weights=weights)) + \
                               ' Avg P|Noobj (R/F): ' \
                               '{:.3f}/{:.3f}, '.format(np.average(batch_stats['avg_rpn_noobj'], weights=weights),
                                                        np.average(batch_stats['avg_frcn_background'], weights=weights)) + \
                               ' Validation Loss (R/F): ' \
                               '{:.5f}/{:.5f}'.format(val_losses[0],
                                                        val_losses[1])
                else:
                    disp_str = ' Training Loss (R/F): ' \
                               '{:.5f}/{:.5f}, '.format(np.average(rpn_losses, weights=weights),
                                                        np.average(frcn_losses, weights=weights)) + \
                               ' Avg IOU (R): {:.3f},  '.format(np.average(batch_stats['avg_rpn_iou'], weights=weights)) + \
                               ' Avg Class (R/F): ' \
                               '{:.3f}/{:.3f},  '.format(np.average(batch_stats['avg_rpn_obj'], weights=weights),
                                                         np.average(batch_stats['avg_frcn_class'], weights=weights)) + \
                               ' Avg P|Noobj (R/F): ' \
                               '{:.3f}/{:.3f}'.format(np.average(batch_stats['avg_rpn_noobj'], weights=weights),
                                                      np.average(batch_stats['avg_frcn_background'], weights=weights))
                    inner.set_postfix_str(disp_str)
                with open('training_loss.txt', 'a') as fl:
                    fl.writelines('Epoch: {} '.format(epoch) + disp_str + '\n')
            if epoch % checkpoint_frequency == 0:
                self.save_model(self.name + '{}.pkl'.format(epoch))

        return all_train_losses, all_train_stats, all_val_losses, all_val_stats

    def calculate_loss(self, data, rpn_batch_size=256, frcn_batch_size=64, stats=None, fraction=0.1):
        """
        Calculates the loss for a random partition of a given dataset without
        tracking gradients. Useful for displaying the validation loss during
        training or the test loss during evaluation.
        Parameters
        ----------
        data : PascalDatasetImage
            A dataset object which returns images and image info to use for calculating
            the loss. Only a fraction of the images in the dataset will be tested.
        fraction : float
            The fraction of images from data that the loss should be calculated for.

        Returns
        -------
        tuple
            The mean RPN and FRCN loss over the fraction of the images that were sampled from
            the data.
        """
        val_dataloader = DataLoader(dataset=data,
                                    shuffle=True,
                                    num_workers=NUM_WORKERS)
        rpn_losses = []
        frcn_losses = []

        if stats is None:
            stats = {'avg_rpn_iou': [],
                     'avg_anch_iou': [],
                     'avg_rpn_obj': [],
                     'avg_frcn_class': [],
                     'avg_rpn_noobj': [],
                     'avg_frcn_background': []}

        with torch.no_grad():
            for i, (images, _, (images_bboxes, images_classes)) in enumerate(val_dataloader):
                images = images.to(self.device)
                images_bboxes = images_bboxes[0].to(self.device)
                images_classes = images_classes[0].to(self.device)
                features = self.fast_rcnn.backbone.features(images)
                reg, cls = self.rpn(features)
                image_size = images.shape[-2:]
                grid_size = features.shape[-2:]
                anchors_xyxy = self.rpn.construct_anchors(image_size, grid_size)
                valid = self.rpn.validate_anchors(anchors_xyxy)  # This has the effect of scrambling anchors
                reg = reg[:, valid]
                cls = cls[:, valid]
                anchors_xyxy = anchors_xyxy[valid]
                anchors_xywh = xyxy2xywh(anchors_xyxy)
                samples, roi_match = self.rpn.sample(rpn_batch_size, images_bboxes, anchors_xyxy, stats)
                loss, stats = self.rpn.loss(reg[:, samples],
                                            cls[:, samples],
                                            images_bboxes[roi_match],
                                            anchors_xywh[samples].detach(),
                                            stats)
                rpn_loss = loss['total']
                rois = self.rpn.post_process(reg, cls, anchors_xywh, PRE_NMS_TOPK_TRAIN, POST_NMS_TOPK_TRAIN, True)[0]
                # FRCN
                samples, roi_match = self.fast_rcnn.sample(frcn_batch_size, images_bboxes, rois, stats)
                rois_xyxy = rois[samples]
                reg, cls = self.fast_rcnn(features, rois_xyxy.detach(), extract_features=False)
                images_bboxes = images_bboxes[roi_match]
                images_classes = images_classes[roi_match]
                rois_xywh = xyxy2xywh(rois_xyxy)
                loss, stats = self.fast_rcnn.loss(reg, cls, images_bboxes, images_classes, rois_xywh.detach(),
                                                  stats)
                frcn_loss = loss['total']
                rpn_losses.append(rpn_loss.item())
                frcn_losses.append(frcn_loss.item())
                if i > len(val_dataloader) * fraction:
                    break

            return [np.mean(rpn_losses), np.mean(frcn_losses)], stats

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

    def get_trainable_parameters(self):
        """
        Returns a list of a model's trainable parameters by checking which
        parameters are tracking their gradients.
        Returns
        -------
        list
            A list containing the trainable parameters.
        """
        trainable_parameters = []
        for param in self.parameters():
            if param.requires_grad:
                trainable_parameters.append(param)

        return trainable_parameters
