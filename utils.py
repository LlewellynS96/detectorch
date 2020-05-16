import os
import random
import cv2
import numpy as np
import xml.etree.ElementTree as Et
import torch
import csv
from functools import reduce
from time import time


NUM_WORKERS = 0

VGG_MEAN = [0.458, 0.438, 0.405]
VGG_STD = [0.247, 0.242, 0.248]

SS_MEAN = [0.174, 0.634, 0.505]
SS_STD = [0.105, 0.068, 0.071]


def read_classes(file):
    """
    Parses a text file containing all the classes that are present
    in a specific dataset.
    Parameters
    ----------
    file : str
        The path to the text file to be read.

    Returns
    -------
    list
        A list containing the classes read from the text file.
    """
    file = open(file, 'r')
    lines = file.read().split('\n')
    lines = [l for l in lines if len(l) > 0]
    classes = [l for l in lines if l[0] != '#']

    return classes


def get_annotations(annotations_dir, img):
    """
    Collects all the annotations for a specific image in any of the
    Pascal VOC datasets.
    Parameters
    ----------
    annotations_dir : str
        The path to the directory containing all the Pascal VOC annotations.
    img : str
        The ID of the image for which the annotations have to be collected.

    Returns
    -------
    list
        A list of tuples containing the annotations for the requested image.
        Each tuple corresponds to a single bounding box and the information
        withing each tuple is (class_name, bbox_xmin, bbox_ymin, bbox_xmax,
        bbox_ymax, is_truncated, is_difficult). The bounding box coordinates
        are normalized to the width and height of the image.

    """
    file = os.path.join(annotations_dir, img + '.xml')
    tree = Et.parse(file)
    root = tree.getroot()

    annotations = []

    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        name = obj.find('name').text
        truncated = int(obj.find('truncated').text)
        difficult = int(obj.find('difficult').text)

        xmin = float(bbox.find('xmin').text)
        xmax = float(bbox.find('xmax').text)
        ymin = float(bbox.find('ymin').text)
        ymax = float(bbox.find('ymax').text)
        annotations.append((name, height, width, xmin, ymin, xmax, ymax, truncated, difficult))

    return annotations


def to_numpy_image(image, size, normalize=True):
    """
    Converts a Tensor in the range [0., 1.] to a resized
    Numpy array in the range [0, 255].
    Parameters
    ----------
    image : Tensor
        A Tensor representing the image.
    size : tuple of int
        The size (w, h) to which the image should be resized.
    normalize : bool
        A flag which indicates whether the image was originally normalized,
        which means that it should be "de-normalized" when converting to an
        array.
    Returns
    -------
    image : ndarray
        A Numpy array representation of the image.
    """
    if isinstance(image, torch.Tensor):
        image = np.array(image.permute(1, 2, 0).cpu().numpy())
    if normalize:
        # image *= VGG_STD
        # image += VGG_MEAN
        image *= SS_STD
        image += SS_MEAN
    image *= 255.
    image = image.astype(dtype=np.uint8)
    image = cv2.resize(image, dsize=size, interpolation=cv2.INTER_CUBIC)

    return image


def add_bbox_to_image(image, bbox, confidence, cls, color=None):
    """
    Adds a visual bounding box with labels to an image in-place.

    Parameters
    ----------
    image : ndarray
        A Numpy array containing the image.
    bbox : array_like
        An array (x1, y1, x2, y2) containing the coordinates of the upper-
        left and bottom-right corners of the bounding box to be added to
        the image. The coordinates should be normalized to the width and
        the height of the image.
    confidence : float
        A value representing the confidence of an object within the bounding
        box. This value will be displayed as part of the label.
    cls : str
        The class to which the object in the bounding box belongs. This
        value will be displayed as part of the label.
    """
    if confidence is not None:
        text = '{} {:.2f}'.format(cls, confidence)
    else:
        text = '{}'.format(cls)
    xmin, ymin, xmax, ymax = bbox
    # Draw a bounding box.
    if color is None:
        color = np.random.uniform(0., 255., size=3)
    cv2.rectangle(image, (xmin, ymax), (xmax, ymin), color, 3)

    # Display the label at the top of the bounding box
    label_size, base_line = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    ymax = max(ymax, label_size[1])
    cv2.rectangle(image,
                  (xmin, ymax - round(1.5 * label_size[1])),
                  (xmin + round(1.5 * label_size[0]),
                   ymax + base_line),
                  color,
                  cv2.FILLED)
    cv2.putText(image, text, (xmin, ymax), cv2.FONT_HERSHEY_SIMPLEX, 0.75, [255] * 3, 1)


def jaccard(boxes_a, boxes_b):
    """
    Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.
    Parameters
    ----------
        boxes_a : Tensor
            An array whose shape is :math:`(N, 4)`. :math:`N` is the number
            of bounding boxes. The dtype should be :obj:`float`.
        boxes_b : Tensor
            An array similar to :obj:`bbox_a`, whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`float`.
    Returns
    -------
        Tensor
            An array whose shape is :math:`(N, K)`. An element at index :math:`(n, k)`
            contains IoUs between :math:`n` th bounding box in :obj:`bbox_a` and
            :math:`k` th bounding box in :obj:`bbox_b`.
    Notes
    -----
        from: https://github.com/chainer/chainercv
    """
    assert boxes_a.shape[1] == 4
    assert boxes_b.shape[1] == 4
    assert isinstance(boxes_a, torch.Tensor)
    assert isinstance(boxes_b, torch.Tensor)

    tl = torch.max(boxes_a[:, None, :2], boxes_b[:, :2])
    br = torch.min(boxes_a[:, None, 2:], boxes_b[:, 2:])

    area_a = torch.prod(boxes_a[:, 2:] - boxes_a[:, :2], 1)
    area_b = torch.prod(boxes_b[:, 2:] - boxes_b[:, :2], 1)

    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en

    # area_a = torch.clamp(area_a, min=0)

    ious = area_i / (area_a[:, None] + area_b - area_i)
    ious[torch.isnan(ious)] = 0.

    return ious


def access_dict_list(dictionary, index):
    """
    Returns the items at a specified index in a dictionary
    as a dictionary. This is used to convert a dictionary of
    lists to a list of dictionaries.
    Parameters
    ----------
    dictionary : dict
        The dictionary containing lists of items.
    index : int
        The index in each list which should be returned.

    Returns
    -------
    dict
        The dictionary contained at the specified index.
    """
    try:
        return {k: v[index] for k,v in dictionary.items() if isinstance(v, (list, torch.Tensor))}
    except IndexError:
        return None


def sample_ids(n, pos_ids, neg_ids, pos_ratio):
    num_neg = len(neg_ids)
    num_pos = len(pos_ids)
    n_p = int(n * pos_ratio)
    ids = torch.zeros(n, dtype=torch.long)
    if num_pos < n_p:
        n_p = num_pos
    ids[:n_p] = pos_ids[torch.randperm(num_pos)[:n_p]].squeeze()
    n_n = n - n_p
    if num_neg < n_n:
        ids[-n_n:] = neg_ids[torch.randint(num_neg, (n_n,))].squeeze()
    else:
        ids[-n_n:] = neg_ids[torch.randperm(num_neg)[:n_n]].squeeze()

    return ids, n_p, n_n


def xyxy2xywh(xyxy):
    """
    Converts bounding boxes that are in the form (x1, y1, x2, y2)
    to (x_c, y_c, w, h).
    Parameters
    ----------
    xyxy : Tensor
        A Tensor whose shape is :math:`(N, 4)` where the elements in the
        0th and 1st column represent the x and y coordinates of the top left
        corner of the bounding box and the 2nd and 3rd column represent the
        x and y coordinates of the bottom right corner of the bounding box.
    Returns
    -------
    xywh : Tensor
        A Tensor whose shape is :math:`(N, 4)` where the elements in the
        0th and 1st column represent the x and y coordinates of the center
        of the bounding box and the 2nd and 3rd column represent the
        width and height of the bounding box.
    """
    xywh = torch.zeros_like(xyxy)
    xywh[:, :2] = (xyxy[:, :2] + xyxy[:, 2:]) / 2.
    xywh[:, 2:] = xyxy[:, 2:] - xyxy[:, :2]

    return xywh


def xywh2xyxy(xywh):
    """
    Converts bounding boxes that are in the form (x_c, y_c, w, h)
    to (x1, y1, x2, y2).
    Parameters
    ----------
    xywh : Tensor
        A Tensor whose shape is :math:`(N, 4)` where the elements in the
        0th and 1st column represent the x and y coordinates of the center
        of the bounding box and the 2nd and 3rd column represent the
        width and height of the bounding box.

    Returns
    -------
    xyxy : Tensor
        A Tensor whose shape is :math:`(N, 4)` where the elements in the
        0th and 1st column represent the x and y coordinates of the top left
        corner of the bounding box and the 2nd and 3rd column represent the
        x and y coordinates of the bottom right corner of the bounding box.
    """
    xyxy = torch.zeros_like(xywh)
    half = xywh[:, 2:] / 2.
    xyxy[:, :2] = xywh[:, :2] - half
    xyxy[:, 2:] = xywh[:, :2] + half

    return xyxy


def parameterize_bboxes(bboxes, anchors_xywh):
    """
    Parameterizes bounding boxes according to the R-CNN conventions
    given the coordinates for the bounding boxes and their
    corresponding anchors.
    Parameters
    ----------
    bboxes : Tensor
        A Tensor whose shape is :math:`(N, 4)` representing the coordinates
        of :math:`N` bounding boxes in the format (x1, y1, x2, y2).
    anchors : Tensor
        A Tensor whose shape is :math:`(N, 4)` representing the coordinates
        of the :math:`N` anchors in the format (x1, y1, x2, y2) that correspond
        to the different bounding boxes.

    Returns
    -------
    Tensor
        A Tensor whose shape is :math:`(N, 4)` representing the R-CNN bounding
        box parameters of :math:`N` bounding boxes in the format
        :math:`(t_x, t_y, t_w, t_h)`.
    """
    assert bboxes.shape == anchors_xywh.shape
    assert bboxes.shape[-1] == 4

    bboxes_xywh = xyxy2xywh(bboxes)

    bboxes_xywh[:, :2] = (bboxes_xywh[:, :2] - anchors_xywh[:, :2]) / anchors_xywh[:, 2:]
    bboxes_xywh[:, 2:] = torch.log(bboxes_xywh[:, 2:] / anchors_xywh[:, 2:])

    return bboxes_xywh


def deparameterize_bboxes(reg, anchors_xywh):
    """
    Parameterizes bounding boxes according to the R-CNN conventions
    given the coordinates for the bounding boxes and their
    corresponding anchors.
    Parameters
    ----------
    reg : Tensor
        A Tensor whose shape is :math:`(N, 4)` representing the R-CNN bounding
        box parameters of :math:`N` bounding boxes in the format
        :math:`(t_x, t_y, t_w, t_h)`.
    anchors_xywh : Tensor
        A Tensor whose shape is :math:`(N, 4)` representing the coordinates
        of the :math:`N` anchors in the format (x, y, w, h) that correspond
        to the different bounding boxes.

    Returns
    -------
    Tensor
        A Tensor whose shape is :math:`(N, 4)` representing the coordinates
        of :math:`N` bounding boxes in the format (x1, y1, x2, y2).
    """
    assert reg.shape == anchors_xywh.shape
    assert reg.shape[-1] == 4

    bboxes_xywh = torch.zeros_like(anchors_xywh)

    bboxes_xywh[:, :2] = reg[:, :2] * anchors_xywh[:, 2:] + anchors_xywh[:, :2]
    bboxes_xywh[:, 2:] = torch.exp(reg[:, 2:]) * anchors_xywh[:, 2:]

    bboxes_xyxy = xywh2xyxy(bboxes_xywh)

    return bboxes_xyxy


def get_trainable_parameters(module):
    """
    Returns a list of a model's trainable parameters by checking which
    parameters are tracking their gradients.
    Returns
    -------
    list
        A list containing the trainable parameters.
    """
    trainable_parameters = []
    for param in module.parameters():
        if param.requires_grad:
            trainable_parameters.append(param)

    return trainable_parameters


def to_repr(x):
    """
    A small utility function that converts floats and strings to
    a fixed string format.
    Parameters
    ----------
    x
        An input for which a string representation should be provided.
    Returns
    -------
    str
        A string representation of the input.

    """
    if isinstance(x, (float, np.float, np.float32)):
        return '{:.6f}'.format(x)
    else:
        return str(x)


def export_prediction(cls, image_id, top, left, bottom, right, confidence,
                      prefix='comp4', set_name='test', directory='detections'):
    """
    Exports a single predicted bounding box to a text file by appending it to a file
    in the format specified by the Pascal VOC competition.
    Parameters
    ----------
    cls : str
        The predicted class name of the specified bounding box.
    image_id : str
        The Pascal VOC image ID, i.e. the image's file name.
    top : float
        The y-coordinate of the top-left corner of the predicted bounding box. The value
        should be normalized to the height of the image.
    left : float
        The x-coordinate of the top-left corner of the predicted bounding box. The value
        should be normalized to the width of the image.
    bottom : float
        The y-coordinate of the bottom-right corner of the predicted bounding box. The value
        should be normalized to the height of the image.
    right : float
        The x-coordinate of the bottom-right corner of the predicted bounding box. The value
        should be normalized to the width of the image.
    confidence : float
        A confidence value attached to the prediction, which should be generated by the
        detector.  The value does not have to be normalized, but a greater value corresponds
        to greater confidence.  This value is used when calculating the precision-recall graph
        for the detector.
    prefix : str
        A string value that is prepended to the file where the predictions are stored.  For
        PASCAL VOC competitions this value is 'comp' + the number of the competition being
        entered into, e.g. comp4.
    set_name : str
        The subset for which the predictions were made, e.g. 'val', 'test' etc..
    directory : str
        The directory to which all the prediction files should be saved.

    Returns
    -------
    None
    """
    filename = prefix + '_det_' + set_name + '_' + cls + '.txt'
    filename = os.path.join(directory, filename)

    with open(filename, 'a') as f:
        prediction = [image_id, confidence, np.round(left), np.round(top), np.round(right), np.round(bottom), '\n']
        prediction = map(to_repr, prediction)
        prediction = ' '.join(prediction)
        f.write(prediction)


def get_trainable_parameters(module):
    """
    Returns a list of a model's trainable parameters by checking which
    parameters are tracking their gradients.
    Returns
    -------
    list
        A list containing the trainable parameters.
    """
    trainable_parameters = []
    for param in module.parameters():
        if param.requires_grad:
            trainable_parameters.append(param)

    return trainable_parameters


class nullcontext:

    def __enter__(self):
        return None

    def __exit__(self, *args):
        pass


def set_random_seed(x):
    np.random.seed(x)
    torch.random.manual_seed(x)
    random.seed(x)


def step_decay_scheduler(optimizer, steps=None, scales=None):
    if steps is None or scales is None:
        steps = [-1, 100]
        scales = [0.1, 10.]

    def foo(e):
        if e < min(steps):
            return 1.
        for i, s in enumerate(reversed(steps)):
            if e >= s:
                return reduce(lambda x, y: x * y, scales[:len(steps) - i])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=foo)

    return scheduler


def find_best_anchors(classes, root_dir, dataset, min_size=600, k=5, max_iter=20, skip_truncated=True, weighted=True, init=600, device='cuda'):

    annotations_dir = [os.path.join(r, 'Annotations') for r in root_dir]
    sets_dir = [os.path.join(r, 'ImageSets', 'Main') for r in root_dir]

    images = []

    for d in range(len(dataset)):
        for cls in classes:
            file = os.path.join(sets_dir[d], '{}_{}.txt'.format(cls, dataset[d]))
            with open(file) as f:
                for line in f:
                    image_desc = line.split()
                    if image_desc[1] == '1':
                        images.append((d, image_desc[0]))

    images = list(set(images))
    bboxes = []

    for image in images:
        annotations = get_annotations(annotations_dir[image[0]], image[1])
        for annotation in annotations:
            name, height, width, xmin, ymin, xmax, ymax, truncated, difficult = annotation
            if skip_truncated and truncated:
                continue
            scale = min_size * 1.0 / min(width, height)
            width = (xmax - xmin) * scale
            height = (ymax - ymin) * scale
            bboxes.append([0., 0., width, height])

    bboxes = torch.tensor(bboxes, device=device)
    anchors = torch.tensor(([0., 0., init, init] * np.random.random((k, 4))).astype(dtype=np.float32), device=device)

    for _ in range(max_iter):
        ious = jaccard(bboxes, anchors)
        iou_max, idx = torch.max(ious, dim=1)
        for i in range(k):
            if weighted:
                weights = (torch.tensor([1.], device=device) - iou_max[idx == i, None]) ** 2
                anchors[i] = torch.sum(bboxes[idx == i] * weights, dim=0) / torch.sum(weights)  # Weighted k-means

            else:
                anchors[i] = torch.mean(bboxes[idx == i], dim=0)  # Normal k-means

        sort = torch.argsort(anchors[:, 2], dim=0)
        anchors = anchors[sort]

    return anchors[:, 2:]


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # classes = read_classes('../../../Data/VOCdevkit/voc.names')
    # a = find_best_anchors(classes,
    #                       k=9,
    #                       max_iter=1000,
    #                       root_dir=['../../../Data/VOCdevkit/VOC2007/', '../../../Data/VOCdevkit/VOC2012/'],
    #                       dataset=['trainval'] * 2,
    #                       min_size=600,
    #                       skip_truncated=False,
    #                       weighted=True,
    #                       device=device)
    classes = read_classes('../../../Data/SS/ss.names')
    a = find_best_anchors(classes,
                          k=9,
                          max_iter=1000,
                          root_dir=['../../../Data/SS/'],
                          dataset=['train'],
                          init=50,
                          min_size=600,
                          skip_truncated=False,
                          weighted=True,
                          device=device)

    for x, y in a:
        print('[{:.0f},{:.0f}], '.format(x, y))


if __name__ == '__main__':
    main()
