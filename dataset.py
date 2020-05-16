import os
import random
import torch
import scipy.signal
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms
from utils import to_numpy_image, add_bbox_to_image
from utils import read_classes, get_annotations
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt

RESIZE = True
# MIN_TRAIN_SIZE = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
# MIN_TEST_SIZE = 800
# MAX_SIZE = 1333
MIN_TRAIN_SIZE = [600]
MIN_TEST_SIZE = 600
MAX_SIZE = 1000
SMALL_THRESHOLD = 0.005


class PascalDatasetImage(Dataset):
    """
    This object can be configured to return images and annotations
    from a PASCAL VOC dataset in a general format that is compatible with
    any arbitrary object detector.
    """
    def __init__(self, classes, mu, sigma, train=False, root_dir='data/VOC2012/', dataset='train', skip_truncated=True,
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

        self.train = train

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
        image = Image.open(os.path.join(self.images_dir[dataset], img + '.jpg'))
        image_info = {'id': img, 'width': image.width, 'height': image.height, 'dataset': self.dataset[dataset]}
        if self.do_transforms:
            random_flip = np.random.random()
            random_blur = np.random.random()
            image = torchvision.transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05)(image)
            if random_flip >= 0.5:
                image = torchvision.transforms.functional.hflip(image)
            if random_blur >= 0.9:
                image = image.filter(ImageFilter.GaussianBlur(radius=1))
        if RESIZE:
            if self.train:
                size = np.random.choice(MIN_TRAIN_SIZE)
            else:
                size = MIN_TEST_SIZE
            scale = size * 1.0 / min(image.width, image.height)
            if image.height < image.width:
                w, h = int(scale * image.width), size
            else:
                w, h = size, int(scale * image.height)
            if max(h, w) > MAX_SIZE:
                scale = MAX_SIZE * 1.0 / max(h, w)
                w = int(w * scale)
                h = int(h * scale)
            image = image.resize((w, h))

        image = torchvision.transforms.ToTensor()(image)
        # ============================= NORMALIZATION =============================
        # Include for VGG models / Exclude for pretrained Faster R-CNN models.
        image = torchvision.transforms.Normalize(mean=self.mu, std=self.sigma)(image)

        if self.return_targets:
            annotations = get_annotations(self.annotations_dir[dataset], image_info['id'])
            bboxes = []
            classes = []
            # For each object in image.
            for annotation in annotations:
                name, height, width, xmin, ymin, xmax, ymax, truncated, difficult = annotation
                xmin /= width
                xmax /= width
                ymin /= height
                ymax /= height
                if (self.skip_truncated and truncated) or (self.skip_difficult and difficult):
                    continue
                if name not in self.classes:
                    continue
                if self.do_transforms:
                    if random_flip >= 0.5:
                        tmp = xmin
                        xmin = 1. - xmax
                        xmax = 1. - tmp
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


class SSDatasetSTFT(Dataset):

    def __init__(self, classes, mu, sigma, root_dir, dataset, train=False, skip_truncated=True,
                 do_transforms=False, skip_difficult=True, return_targets=False, mode='spectrogram_db'):

        self.classes = read_classes(classes)
        self.classes.insert(0, '__background__')

        self.num_classes = len(self.classes)

        if isinstance(root_dir, str):
            root_dir = [root_dir]
        if isinstance(dataset, str):
            dataset = [dataset]

        assert len(root_dir) == len(dataset)

        self.root_dir = root_dir
        self.raw_dir = [os.path.join(r, 'Raw') for r in self.root_dir]
        self.annotations_dir = [os.path.join(r, 'Annotations') for r in self.root_dir]
        self.sets_dir = [os.path.join(r, 'ImageSets', 'Main') for r in self.root_dir]
        self.dataset = dataset

        self.train = train

        self.mode = mode

        self.mu = mu
        self.sigma = sigma

        self.data = []
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
                            self.data.append((d, image_desc[0]))

        self.data = list(set(self.stfts))  # Remove duplicates.
        self.data.sort()
        # random.shuffle(self.data)

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
        dataset, img = self.data[index]
        data = np.load(os.path.join(self.raw_dir[dataset], img + '.npz'))
        signal = data['signal']
        samp_rate = data['samp_rate']
        N_fft = data['N_fft']
        N_overlap = data['N_overlap']
        signal = signal[0] + 1.j * signal[1]
        stft, _, _ = self.stft(signal,
                               N_fft=N_fft,
                               N_overlap=N_overlap,
                               samp_rate=samp_rate)
        if self.mode == 'spectrogram':
            data = np.abs(stft) ** 2
        elif self.mode == 'spectrogram_db':
            data = 10. * np.log10(np.abs(stft) ** 2)
        elif self.mode == 'stft_iq':
            data = [stft.real, stft.imag]
        elif self.mode == 'stft_ap':
            data = [np.abs(stft), np.angle(stft)]
        else:
            raise ValueError('Unknown mode. Use one of spectrogram, spectrogram_db, stft_iq, or stft_ap.')

        data = torch.tensor(data, dtype=torch.float32)

        data_info = {'id': img, 'width': data.shape[1], 'height': data.shape[0], 'dataset': self.dataset[dataset]}

        if self.return_targets:
            annotations = get_annotations(self.annotations_dir[dataset], data_info['id'])
            bboxes = []
            classes = []
            # For each object in image.
            for annotation in annotations:
                name, height, width, xmin, ymin, xmax, ymax, truncated, difficult = annotation
                xmin /= width
                xmax /= width
                ymin /= height
                ymax /= height
                if (self.skip_truncated and truncated) or (self.skip_difficult and difficult):
                    continue
                if name not in self.classes:
                    continue
                if self.do_transforms:
                    pass
                xmin = np.clip(xmin, a_min=0, a_max=1)
                xmax = np.clip(xmax, a_min=0, a_max=1)
                ymin = np.clip(ymin, a_min=0, a_max=1)
                ymax = np.clip(ymax, a_min=0, a_max=1)
                if (xmax - xmin) > SMALL_THRESHOLD and (ymax - ymin) > SMALL_THRESHOLD:
                    bboxes.append([xmin, ymin, xmax, ymax])
                    classes.append(self.encode_categorical(name))

            bboxes = torch.tensor(bboxes, dtype=torch.float)
            classes = torch.tensor(classes, dtype=torch.float)

            return data, data_info, (bboxes, classes)
        else:
            return data, data_info

    def __len__(self):

        return len(self.data)

    def encode_categorical(self, name):
        y = self.classes.index(name)
        yy = np.zeros(self.num_classes)
        yy[y] = 1
        return yy

    def decode_categorical(self, one_hot):
        idx = np.argmax(one_hot, axis=-1)
        return self.classes[idx]

    @staticmethod
    def stft(x, N_fft=512, N_overlap=64, samp_rate=10e6):
        f, t, specgram = scipy.signal.stft(x,
                                           fs=samp_rate,
                                           nperseg=N_fft,
                                           noverlap=N_overlap,
                                           return_onesided=False,
                                           boundary=None,
                                           padded=False)
        idx = np.argsort(f)
        specgram = specgram[idx]
        f = f[idx]

        return specgram, f, t


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    db = SSDatasetSTFT(classes='../../../Data/SS/ss.names',
                       root_dir='../../../Data/SS/',
                       dataset='train',
                       mode='stft_iq',
                       mu=[0., 0., 0.],
                       sigma=[1., 1., 1.])

    # db = PascalDatasetImage(classes='../../../Data/VOCdevkit/voc.names',
    #                         root_dir='../../../Data/VOCdevkit/VOC2007/',
    #                         dataset='train',
    #                         mu=[0., 0., 0.],
    #                         sigma=[1., 1., 1.])

    dataloader = DataLoader(dataset=db,
                            shuffle=False,
                            num_workers=1)

    for _ in dataloader:
        print('Ow')