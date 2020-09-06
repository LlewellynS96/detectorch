import os
import pickle
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from utils import to_numpy_image, add_bbox_to_image, get_annotations, read_classes
import torch
import torchvision.transforms
from detectron import FasterRCNN
from dataset import *
from tqdm import tqdm
import scipy.signal
import gizeh


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
        elif self.mode == 'spectrogram_ap':
            data = [np.abs(stft) ** 2, np.angle(stft)]
        elif self.mode == 'spectrogram_ap_db':
            data = [10. * np.log10(np.abs(stft) ** 2), np.angle(stft)]
        elif self.mode == 'stft_iq':
            data = [stft.real, stft.imag]
        elif self.mode == 'stft_ap':
            data = [np.abs(stft), np.angle(stft)]
        else:
            raise ValueError('Unknown mode. Use one of spectrogram, spectrogram_db, '
                             'spectrogram_ap, spectrogram_ap_db, stft_iq or stft_ap.')

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


def create_pascal_label_colormap(num_classes=21):
    """
    Creates a label colormap used in PASCAL VOC segmentation benchmark.
    Returns

    A colormap for visualizing segmentation results.
    """
    def bit_get(val, idx):
        """
        Gets the bit value.
        Parameters
        ----------
        val: int or numpy int array
            Input value.
        idx:
            Which bit of the input val.
        Returns
        -------
        The "idx"-th bit of input val.
        """
        return (val >> idx) & 1

    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= bit_get(ind, channel) << shift
        ind >>= 3

    return colormap[:num_classes] / 255.


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
        image = to_numpy_image(image, size=(image_info['width'], image_info['height']))
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


def show_gt():
    # dataset = ImageDataset(root_dir='../../../Data/VOCdevkit/VOC2007/',
    #                              classes='../../../Data/VOCdevkit/voc.names',
    #                              dataset='test'
    #                              )
    dataset =ImageDataset(root_dir='../../../Data/AMC/',
                          classes='../../../Data/AMC/amc.names',
                          dataset='train'
                          )

    for image, image_info, (bboxes, classes) in dataset:
        image = to_numpy_image(image, size=(image_info['width'], image_info['height']))
        for bbox, cls, difficult, truncated in zip(bboxes, classes, image_info['difficult'], image_info['truncated']):
            name = dataset.classes[cls]
            xmin, ymin, xmax, ymax = bbox
            xmin *= image_info['width']
            ymin *= image_info['height']
            xmax *= image_info['width']
            ymax *= image_info['height']
            bbox = [int(xmin), int(ymin), int(xmax), int(ymax)]
            add_bbox_to_image(image, bbox, None, name, 2, [0., 255., 0])
        plt.imshow(image)
        plt.axis('off')
        plt.show()


def calculate_mu_sigma(root_dir, classes, dataset, ndims):
    # dataset = DefaultImageDataset(root_dir=root_dir,
    #                               classes=classes,
    #                               dataset=dataset
    #                               )
    dataset = SSDataset(root_dir=root_dir,
                        classes=classes,
                        dataset=dataset,
                        mode='stft_ap'
                        )

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


def draw_vector_bboxes(model, data, image_size=512, threshold=.8):
    colors = create_pascal_label_colormap(data.num_classes-1)

    bboxes, classes, confidence, image_idx = model.predict(dataset=data,
                                                           confidence_threshold=threshold,
                                                           overlap_threshold=.3,
                                                           show=False,
                                                           export=False
                                                           )

    for idx in np.unique(image_idx):
        image = Image.open(os.path.join(data.root_dir[0], 'JPEGImages', idx + '.jpg'))
        image = image.resize((image_size, image_size))
        image = gizeh.ImagePattern(np.array(image))
        image = gizeh.rectangle(2*image_size,
                                2*image_size,
                                xy=(0, 0),
                                fill=image)
        pdf = gizeh.PDFSurface('detections/{}.pdf'.format(idx),
                               image_size,
                               image_size)
        image.draw(pdf)
        mask = np.array(image_idx) == idx
        _bboxes = bboxes[mask]
        _classes = classes[mask]
        _confidence = confidence[mask]
        argsort_x = torch.argsort(_bboxes[:, 0])
        argsort_y = torch.argsort(_bboxes[argsort_x][:, 1])
        _bboxes = _bboxes[argsort_x][argsort_y]
        _classes = _classes[argsort_x][argsort_y]
        _confidence = _confidence[argsort_x][argsort_y]
        for bb, cl, co in zip(_bboxes, _classes, _confidence):
            rect = [[int(bb[0]), int(bb[1])],
                    [int(bb[2]), int(bb[1])],
                    [int(bb[2]), int(bb[3])],
                    [int(bb[0]), int(bb[3])]]
            rect = gizeh.polyline(rect, close_path=True, stroke_width=4, stroke=colors[cl-1])
            rect.draw(pdf)
        for bb, cl, co in zip(_bboxes, _classes, _confidence):
            w, h = len(data.classes[cl]) * 8.5 + 65, 15
            rect = gizeh.rectangle(w,
                                   h,
                                   xy=(int(bb[0] + w / 2 - 2),
                                       int(bb[1] - h / 2 + 7)),
                                   fill=(1, 1, 1, 0.5))

            rect.draw(pdf)
            txt = gizeh.text('{}: {:.2f}'.format(data.classes[cl], co),
                             'monospace',
                             fontsize=16,
                             xy=(int(bb[0]),
                                 int(bb[1])),  # - 12),
                             fill=(0., 0., 0.),
                             v_align='center',
                             h_align='left')
            txt.draw(pdf)
        pdf.flush()
        pdf.finish()


def main():
    # show_difficult()
    show_gt()
    # print(calculate_mu_sigma(root_dir='../../../Data/SS/',
    #                          classes='../../../Data/SS/ss.names',
    #                          dataset='train',
    #                          ndims=2))
    data = PascalDataset(root_dir='../../../Data/SS/',
                         classes='../../../Data/SS/ss.names',
                         dataset='test',
                         skip_difficult=False,
                         skip_truncated=False,
                         mu=[0.174, 0.634, 0.505],
                         sigma=[0.105, 0.068, 0.071],
                         do_transforms=False
                         )
    data.n = 10
    model = pickle.load(open('FasterRCNN18.pkl', 'rb'))

    draw_vector_bboxes(model, data)


if __name__ == '__main__':
    main()
