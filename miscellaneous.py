import os
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from utils import to_numpy_image, add_bbox_to_image, get_annotations, read_classes
import torch
import torchvision.transforms
from tqdm import tqdm
import scipy.signal


class ImageDataset:

    def __init__(self, classes, root_dir, dataset):

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

        self.images = []

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

        annotations = get_annotations(self.annotations_dir[dataset], image_info['id'])
        bboxes = []
        classes = []
        image_info['difficult'] = []
        image_info['truncated'] = []
        # For each object in image.
        for annotation in annotations:
            name, height, width, xmin, ymin, xmax, ymax, truncated, difficult = annotation
            image_info['difficult'].append(difficult)
            image_info['truncated'].append(truncated)
            xmin /= width
            xmax /= width
            ymin /= height
            ymax /= height
            if name not in self.classes:
                continue
            xmin = np.clip(xmin, a_min=0, a_max=1)
            xmax = np.clip(xmax, a_min=0, a_max=1)
            ymin = np.clip(ymin, a_min=0, a_max=1)
            ymax = np.clip(ymax, a_min=0, a_max=1)
            if (xmax - xmin) > 0. and (ymax - ymin) > 0.:
                bboxes.append([xmin, ymin, xmax, ymax])
                classes.append(self.classes.index(name))

        image = torchvision.transforms.ToTensor()(image)

        return image, image_info, (bboxes, classes)

    def __len__(self):
        return len(self.images)


class SSDataset:

    def __init__(self, classes, root_dir, dataset, mode='spectrogram_db'):

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

        self.mode = mode

        self.data = []

        for d in range(len(dataset)):
            for cls in self.classes[1:]:
                file = os.path.join(self.sets_dir[d], '{}_{}.txt'.format(cls, dataset[d]))
                with open(file) as f:
                    for line in f:
                        image_desc = line.split()
                        if image_desc[1] == '1':
                            self.data.append((d, image_desc[0]))

        self.data = list(set(self.data))  # Remove duplicates.
        self.data.sort()
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
        if data.ndim == 2:
            data = data[None]

        data_info = {'id': img, 'width': data.shape[1], 'height': data.shape[0], 'dataset': self.dataset[dataset]}

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
            if name not in self.classes:
                continue
            xmin = np.clip(xmin, a_min=0, a_max=1)
            xmax = np.clip(xmax, a_min=0, a_max=1)
            ymin = np.clip(ymin, a_min=0, a_max=1)
            ymax = np.clip(ymax, a_min=0, a_max=1)
            if (xmax - xmin) > 0. and (ymax - ymin) > 0.:
                bboxes.append([xmin, ymin, xmax, ymax])
                classes.append(self.encode_categorical(name))

        bboxes = torch.tensor(bboxes, dtype=torch.float)
        classes = torch.tensor(classes, dtype=torch.float)

        return data, data_info, (bboxes, classes)

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


def show_difficult():
    # dataset = ImageDataset(root_dir='../../../Data/VOCdevkit/VOC2007/',
    #                              classes='../../../Data/VOCdevkit/voc.names',
    #                              dataset='test'
    #                              )
    dataset =ImageDataset(root_dir='../../../Data/SS/',
                          classes='../../../Data/SS/ss.names',
                          dataset='test'
                          )

    for image, image_info, (bboxes, classes) in dataset:
        image = to_numpy_image(image, size=(image_info['width'], image_info['height']), normalize=False)
        for bbox, cls, difficult, truncated in zip(bboxes, classes, image_info['difficult'], image_info['truncated']):
            name = dataset.classes[cls]
            xmin, ymin, xmax, ymax = bbox
            xmin *= image_info['width']
            ymin *= image_info['height']
            xmax *= image_info['width']
            ymax *= image_info['height']
            bbox = [int(xmin), int(ymin), int(xmax), int(ymax)]
            if difficult:
                add_bbox_to_image(image, bbox, None, name, [255., 0., 0])
            else:
                add_bbox_to_image(image, bbox, None, name, [0., 255., 0])
        plt.imshow(image)
        # plt.axis('off')
        plt.show()


def calculate_mu_sigma():
    # dataset = DefaultImageDataset(root_dir=['../../../Data/VOCdevkit/VOC2007/',
    #                                         '../../../Data/VOCdevkit/VOC2012/'],
    #                               classes='../../../Data/VOCdevkit/voc.names',
    #                               dataset=['trainval'] * 2
    #                               )
    dataset = SSDataset(root_dir='../../../Data/SS/',
                        classes='../../../Data/SS/ss.names',
                        dataset='test',
                        mode='stft_iq'
                        )

    ndims = 2
    mu = torch.zeros(len(dataset), ndims)
    sigma = torch.zeros(len(dataset), ndims)
    i = 0
    for image, _, _ in tqdm(dataset):
        mu[i] = torch.mean(image, dim=(1, 2))
        sigma[i] = torch.std(image, dim=(1, 2))
        i += 1
    mu = torch.mean(mu, dim=0)
    sigma = torch.mean(sigma, dim=0)
    return mu, sigma


def main():
    # show_difficult()
    print(calculate_mu_sigma())


if __name__ == '__main__':
    main()
