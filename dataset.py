import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms
from utils import VGG_MEAN, VGG_STD, SS_MEAN, SS_STD, to_numpy_image, add_bbox_to_image
from utils import read_classes, get_pascal_annotations, get_ss_annotations
from PIL import Image, ImageFilter

SMALL_THRESHOLD = 0.005


class PascalDatasetImage(Dataset):
    """
    This object can be configured to return images and annotations
    from a PASCAL VOC dataset in a general format that is compatible with
    any arbitrary object detector.
    """
    def __init__(self, classes, mu, sigma, root_dir='data/VOC2012/', dataset='train', skip_truncated=True,
                 do_transforms=False, skip_difficult=True, return_targets=False):
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
        """
        self.classes = read_classes(classes)
        self.classes.insert(0, '__background__')

        self.num_classes = len(self.classes)

        if isinstance(root_dir, str):
            root_dir = [root_dir]
        if isinstance(dataset, str):
            dataset = [dataset]

        assert len(root_dir) == len(dataset)

        self.root_dir = root_dir
        self.images_dir = [os.path.join(r, 'JPEGImages') for r in self.root_dir]
        self.annotations_dir = [os.path.join(r, 'Annotations') for r in self.root_dir]
        self.sets_dir = [os.path.join(r, 'ImageSets', 'Main') for r in self.root_dir]
        self.dataset = dataset

        self.mu = mu
        self.sigma = sigma

        self.images = []
        self.skip_truncated = skip_truncated
        self.skip_difficult = skip_difficult

        self.do_transforms = do_transforms
        self.return_targets = return_targets

        for d in range(len(dataset)):
            for cls in self.classes[1:]:
                file = os.path.join(self.sets_dir[d], '{}_{}.txt'.format(cls, dataset[d]))
                with open(file) as f:
                    for line in f:
                        image_desc = line.split()
                        if image_desc[1] == '1':
                            self.images.append((d, image_desc[0]))

        self.images = list(set(self.images))  # Remove duplicates.
        self.images.sort()
        # random.shuffle(self.images)

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
        dataset, img = self.images[index]
        # dataset, img = self.images[index % 3 * 9 + 233]
        # dataset, img = (0, '000026') if index % 2 else (0, '000035')
        # dataset, img = random.choice([(1, '2007_001558'), (1, '2007_000272'), (1, '2007_000999')])
        image = Image.open(os.path.join(self.images_dir[dataset], img + '.jpg'))
        image_info = {'id': img, 'width': image.width, 'height': image.height, 'dataset': self.dataset[dataset]}
        if self.do_transforms:
            random_flip = np.random.random()
            random_blur = np.random.random()
            image = torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)(image)
            image = torchvision.transforms.RandomGrayscale(p=0.1)(image)
            oversize = 0.2
            image_resize = np.array([image_info['width'], image_info['height']] * np.array((1. + oversize)),
                                    dtype=np.int)
            crop_offset = np.random.random(size=1) * oversize * [image_info['width'], image_info['height']]
            crop_offset = crop_offset.astype(dtype=np.int)
            image = image.resize(image_resize)
            if random_flip >= 0.5:
                image = torchvision.transforms.functional.hflip(image)
            if random_blur >= 0.9:
                image = image.filter(ImageFilter.GaussianBlur(radius=1))
            image = torchvision.transforms.functional.crop(image,
                                                           *crop_offset[::-1],
                                                           image_info['height'], image_info['width'])

        image = torchvision.transforms.ToTensor()(image)
        # ============================= NORMALIZATION =============================
        # Include for VGG models / Exclude for pretrained Faster R-CNN models.
        image = torchvision.transforms.Normalize(mean=self.mu, std=self.sigma)(image)

        if self.return_targets:
            annotations = get_pascal_annotations(self.annotations_dir[dataset], image_info['id'])
            bboxes = []
            classes = []
            # For each object in image.
            for annotation in annotations:
                name, xmin, ymin, xmax, ymax, truncated, difficult = annotation
                if (self.skip_truncated and truncated) or (self.skip_difficult and difficult):
                    continue
                if name not in self.classes:
                    continue
                if self.do_transforms:
                    if random_flip >= 0.5:
                        tmp = xmin
                        xmin = 1. - xmax
                        xmax = 1. - tmp
                    xmin = (xmin * image_resize[0] - crop_offset[0]) / image_info['width']
                    xmax = (xmax * image_resize[0] - crop_offset[0]) / image_info['width']
                    ymin = (ymin * image_resize[1] - crop_offset[1]) / image_info['height']
                    ymax = (ymax * image_resize[1] - crop_offset[1]) / image_info['height']
                xmin = np.clip(xmin, a_min=0, a_max=1)
                xmax = np.clip(xmax, a_min=0, a_max=1)
                ymin = np.clip(ymin, a_min=0, a_max=1)
                ymax = np.clip(ymax, a_min=0, a_max=1)
                if (xmax - xmin) > SMALL_THRESHOLD and (ymax - ymin) > SMALL_THRESHOLD:
                    bboxes.append([xmin, ymin, xmax, ymax])
                    classes.append(self.encode_categorical(name))

            bboxes = torch.tensor(bboxes, dtype=torch.float)
            classes = torch.tensor(classes, dtype=torch.float)

            return image, image_info, (bboxes, classes)
        else:
            return image, image_info

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
