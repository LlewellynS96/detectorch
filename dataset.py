import os
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms
from utils import VGG_MEAN, VGG_STD, SS_MEAN, SS_STD
from utils import read_classes, get_pascal_annotations, get_ss_annotations
from PIL import Image


class PascalDatasetImage(Dataset):
    """
    This object can be configured to return images and annotations
    from a PASCAL VOC dataset in a general format that is compatible with
    any arbitrary object detector.
    """
    def __init__(self, classes, mu, sigma, root_dir='data/VOC2012/', dataset='train', skip_truncated=True,
                 do_transforms=False, skip_difficult=True, image_size=(416, 416)):
        """
        Initialise the dataset object with some network and dataset specific parameters.

        Parameters
        ----------
        classes : str
                The path to a text file containing the names of the different classes that
                should be loaded.
        root_dir : str, optional
                The root directory where the PASCAL VOC images and annotations are stored.
        dataset : {'train', 'val', 'trainval', 'test}, optional
                The specific subset of the PASCAL VOC challenge which should be loaded.
        skip_truncated : bool,  optional
                A boolean value to specify whether bounding boxes should be skipped or
                returned for objects that are truncated.
        do_transforms : bool, optional
                A boolean value to determine whether default image augmentation transforms
                should be randomly applied to images.
        skip_difficult : bool,  optional
                A boolean value to specify whether bounding boxes should be skipped or
                returned for objects that have been labeled as 'difficult'.
        image_size : tuple of int, optional
                A tuple (w, h) describing the desired width and height of the images to
                be returned.
        """
        self.classes = read_classes(classes)
        self.classes.insert(0, '__background__')

        assert set(self.classes).issubset({'__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                                           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                                           'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'})

        assert dataset in ['train', 'val', 'trainval', 'test']

        self.num_classes = len(self.classes)

        self.mu = mu
        self.sigma = sigma

        self.root_dir = root_dir
        self.images_dir = os.path.join(self.root_dir, 'JPEGImages/')
        self.annotations_dir = os.path.join(self.root_dir, 'Annotations')
        self.sets_dir = os.path.join(self.root_dir, 'ImageSets', 'Main')
        self.dataset = dataset

        self.images = []
        self.skip_truncated = skip_truncated
        self.skip_difficult = skip_difficult
        self.default_image_size = image_size
        self.image_size = self.default_image_size

        self.do_transforms = do_transforms

        for cls in self.classes[1:]:
            file = os.path.join(self.sets_dir, '{}_{}.txt'.format(cls, dataset))
            with open(file) as f:
                for line in f:
                    image_desc = line.split()
                    if image_desc[1] == '1':
                        self.images.append(image_desc[0])

        self.images = list(set(self.images))  # Remove duplicates.
        self.images.sort()

    def __getitem__(self, index):
        """
        Return some image with its meta information and labeled annotations.

        Parameters
        ----------
        index : int
            The index of the image to be returned.

        Returns
        -------
        image : Tensor
            The image at self.images[index] after some optional transforms have been
            performed as an (w, h, 3) Tensor in the range [0., 1.].
        image_info : dict
            A dictionary object containing meta information about the image.
        target : Tensor
            A Tensor representing the target output of the R-CNN network which was
            used to initialise the dataset object.

        """
        img = self.images[index]
        # img = self.images[0 if index % 2 else 4]
        image = Image.open(os.path.join(self.images_dir, img + '.jpg'))
        image_info = {'id': img,
                      'width': image.width,
                      'height': image.height,
                      'dataset': self.dataset}
        image_transforms = {'horizontal_flip': 0,
                            'corner_crop_h': 0,
                            'corner_crop_v': 0,
                            'image_oversize_h': self.image_size[0],
                            'image_oversize_v': self.image_size[1]}
        max_oversize = 0.1
        horizontal_flip = np.random.random()
        corner_crop_h, corner_crop_v = (np.random.random(size=2) * max_oversize * self.image_size).astype(dtype=np.int)
        if self.do_transforms:
            image = torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.25, hue=0.05)(image)
            image_oversize_h, image_oversize_v = np.array(self.image_size * np.array((1. + max_oversize)), dtype=np.int)
            image = image.resize([image_oversize_h, image_oversize_v])
            image = torchvision.transforms.functional.crop(image,
                                                           corner_crop_v, corner_crop_h,
                                                           *self.image_size[::-1])

            image_transforms['corner_crop_h'] = corner_crop_h
            image_transforms['corner_crop_v'] = corner_crop_v
            image_transforms['image_oversize_h'] = image_oversize_h
            image_transforms['image_oversize_v'] = image_oversize_v

            if horizontal_flip >= 0.5:
                image = torchvision.transforms.functional.hflip(image)
                image_transforms['horizontal_flip'] = 1
        else:
            image = image.resize(self.image_size)

        image = torchvision.transforms.ToTensor()(image)
        # ============================= NORMALIZATION =============================
        # Include for VGG models / Exclude for pretrained Faster R-CNN models.
        image = torchvision.transforms.Normalize(mean=VGG_MEAN, std=VGG_STD)(image)

        return image, image_info, image_transforms

    def __len__(self):

        return len(self.images)

    def encode_categorical(self, name):
        y = self.classes.index(name)
        yy = np.zeros(self.num_classes)
        yy[y] = 1
        return yy

    def decode_categorical(self, one_hot):
        idx = np.argmax(one_hot, axis=-1)
        return self.classes[idx]

    def set_image_size(self, x, y):
        self.image_size = x, y

    def reset_image_size(self):
        self.image_size = self.default_image_size

    def get_gt_bboxes(self, image_info, image_transforms):
        annotations = get_pascal_annotations(self.annotations_dir, image_info['id'])
        bboxes = []
        classes = []
        # For each object in image.
        for annotation in annotations:
            name, xmin, ymin, xmax, ymax, _, _ = annotation
            if (self.skip_truncated and annotation[5]) or (self.skip_difficult and annotation[6]):
                continue
            if name not in self.classes:
                continue
            image_oversize_h = image_transforms['image_oversize_h'].float()
            image_oversize_v = image_transforms['image_oversize_h'].float()
            corner_crop_h = image_transforms['corner_crop_h'].float()
            corner_crop_v = image_transforms['corner_crop_v'].float()

            xmin = (xmin * image_oversize_h - corner_crop_h) / self.image_size[0]
            xmax = (xmax * image_oversize_h - corner_crop_h) / self.image_size[0]
            ymin = (ymin * image_oversize_v - corner_crop_v) / self.image_size[1]
            ymax = (ymax * image_oversize_v - corner_crop_v) / self.image_size[1]
            if image_transforms['horizontal_flip']:
                tmp = xmin
                xmin = 1. - xmax
                xmax = 1. - tmp
            xmin = np.clip(xmin, a_min=0, a_max=1.)
            xmax = np.clip(xmax, a_min=0, a_max=1.)
            ymin = np.clip(ymin, a_min=0, a_max=1.)
            ymax = np.clip(ymax, a_min=0, a_max=1.)
            if (xmax - xmin) > 0 and (ymax - ymin) > 0:
                bboxes.append([xmin, ymin, xmax, ymax])
                classes.append(self.encode_categorical(name))

        bboxes = torch.tensor(bboxes, dtype=torch.float)
        classes = torch.tensor(classes, dtype=torch.float)

        return bboxes, classes


class SignalSeparationDatasetImage(Dataset):
    """
    This object can be configured to return images and annotations
    from a Signal Separation dataset in a general format that is compatible with
    any arbitrary object detector.
    """
    def __init__(self, classes, root_dir='data/VOC2012/', dataset='train', skip_truncated=True,
                 do_transforms=False, skip_difficult=True, image_size=(416, 416)):
        """
        Initialise the dataset object with some network and dataset specific parameters.

        Parameters
        ----------
        classes : str
                The path to a text file containing the names of the different classes that
                should be loaded.
        root_dir : str, optional
                The root directory where the PASCAL VOC images and annotations are stored.
        dataset : {'train', 'val', 'trainval', 'test}, optional
                The specific subset of the PASCAL VOC challenge which should be loaded.
        skip_truncated : bool,  optional
                A boolean value to specify whether bounding boxes should be skipped or
                returned for objects that are truncated.
        do_transforms : bool, optional
                A boolean value to determine whether default image augmentation transforms
                should be randomly applied to images.
        skip_difficult : bool,  optional
                A boolean value to specify whether bounding boxes should be skipped or
                returned for objects that have been labeled as 'difficult'.
        image_size : tuple of int, optional
                A tuple (w, h) describing the desired width and height of the images to
                be returned.
        """
        self.classes = read_classes(classes)
        self.classes.insert(0, '__background__')

        assert set(self.classes).issubset({'__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                                           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                                           'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'})

        assert dataset in ['train', 'val', 'trainval', 'test']

        self.num_classes = len(self.classes)

        self.root_dir = root_dir
        self.images_dir = os.path.join(self.root_dir, 'Spectrograms')
        self.annotations_dir = os.path.join(self.root_dir, 'Annotations')
        self.sets_dir = os.path.join(self.root_dir, 'Sets')
        self.dataset = dataset

        self.images = []
        self.skip_truncated = skip_truncated
        self.skip_difficult = skip_difficult
        self.default_image_size = image_size
        self.image_size = self.default_image_size

        self.do_transforms = do_transforms

        for cls in self.classes[1:]:
            file = os.path.join(self.sets_dir, '{}_{}.txt'.format(cls, dataset))
            with open(file) as f:
                for line in f:
                    image_desc = line.split()
                    if image_desc[1] == '1':
                        self.images.append(image_desc[0])

        self.images = list(set(self.images))  # Remove duplicates.
        self.images.sort()

    def __getitem__(self, index):
        """
        Return some image with its meta information and labeled annotations.

        Parameters
        ----------
        index : int
            The index of the image to be returned.

        Returns
        -------
        image : Tensor
            The image at self.images[index] after some optional transforms have been
            performed as an (w, h, 3) Tensor in the range [0., 1.].
        image_info : dict
            A dictionary object containing meta information about the image.
        target : Tensor
            A Tensor representing the target output of the R-CNN network which was
            used to initialise the dataset object.

        """
        img = self.images[index]
        image = Image.open(os.path.join(self.images_dir, img + '.jpg'))
        image_info = {'id': img,
                      'width': image.width,
                      'height': image.height,
                      'dataset': self.dataset}
        image_transforms = {'horizontal_flip': 0,
                            'corner_crop_h': 0,
                            'corner_crop_v': 0,
                            'image_oversize_h': self.image_size[0],
                            'image_oversize_v': self.image_size[1]}
        max_oversize = 0.1
        horizontal_flip = np.random.random()
        corner_crop_h, corner_crop_v = (np.random.random(size=2) * max_oversize * self.image_size).astype(dtype=np.int)
        if self.do_transforms:
            image = torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.25, hue=0.05)(image)
            image_oversize_h, image_oversize_v = np.array(self.image_size * np.array((1. + max_oversize)), dtype=np.int)
            image = image.resize([image_oversize_h, image_oversize_v])
            image = torchvision.transforms.functional.crop(image,
                                                           corner_crop_v, corner_crop_h,
                                                           *self.image_size[::-1])

            image_transforms['corner_crop_h'] = corner_crop_h
            image_transforms['corner_crop_v'] = corner_crop_v
            image_transforms['image_oversize_h'] = image_oversize_h
            image_transforms['image_oversize_v'] = image_oversize_v

            if horizontal_flip >= 0.5:
                image = torchvision.transforms.functional.hflip(image)
                image_transforms['horizontal_flip'] = 1
        else:
            image = image.resize(self.image_size)

        image = torchvision.transforms.ToTensor()(image)
        # ============================= NORMALIZATION =============================
        # Include for VGG models / Exclude for pretrained Faster R-CNN models.
        image = torchvision.transforms.Normalize(mean=SS_MEAN, std=SS_STD)(image)

        return image, image_info, image_transforms

    def __len__(self):

        return len(self.images)

    def encode_categorical(self, name):
        y = self.classes.index(name)
        yy = np.zeros(self.num_classes)
        yy[y] = 1
        return yy

    def decode_categorical(self, one_hot):
        idx = np.argmax(one_hot, axis=-1)
        return self.classes[idx]

    def set_image_size(self, x, y):
        self.image_size = x, y

    def reset_image_size(self):
        self.image_size = self.default_image_size

    def get_gt_bboxes(self, image_info, image_transforms):
        annotations = get_ss_annotations(self.annotations_dir, image_info['id'])
        bboxes = []
        classes = []
        # For each object in image.
        for annotation in annotations:
            name, xmin, ymin, xmax, ymax, _, _ = annotation
            if (self.skip_truncated and annotation[5]) or (self.skip_difficult and annotation[6]):
                continue
            if name not in self.classes:
                continue
            image_oversize_h = image_transforms['image_oversize_h'].float()
            image_oversize_v = image_transforms['image_oversize_h'].float()
            corner_crop_h = image_transforms['corner_crop_h'].float()
            corner_crop_v = image_transforms['corner_crop_v'].float()

            xmin = (xmin * image_oversize_h - corner_crop_h) / self.image_size[0]
            xmax = (xmax * image_oversize_h - corner_crop_h) / self.image_size[0]
            ymin = (ymin * image_oversize_v - corner_crop_v) / self.image_size[1]
            ymax = (ymax * image_oversize_v - corner_crop_v) / self.image_size[1]
            if image_transforms['horizontal_flip']:
                tmp = xmin
                xmin = 1. - xmax
                xmax = 1. - tmp
            xmin = np.clip(xmin, a_min=0, a_max=1.)
            xmax = np.clip(xmax, a_min=0, a_max=1.)
            ymin = np.clip(ymin, a_min=0, a_max=1.)
            ymax = np.clip(ymax, a_min=0, a_max=1.)
            if (xmax - xmin) > 0 and (ymax - ymin) > 0:
                bboxes.append([xmin, ymin, xmax, ymax])
                classes.append(self.encode_categorical(name))

        bboxes = torch.tensor(bboxes, dtype=torch.float)
        classes = torch.tensor(classes, dtype=torch.float)

        return bboxes, classes
