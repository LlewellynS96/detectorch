import os
import cv2
import numpy as np
import xml.etree.ElementTree as Et
import torch


NUM_WORKERS = 0
VGG_MEAN = [0.485, 0.456, 0.406]
VGG_STD = [0.229, 0.224, 0.225]


def read_classes(file):
    """
    Utility function that parses a text file containing all the classes
    that are present in a specific dataset.
    Parameters
    ----------
    file : str
        A string pointing to the text file to be read.

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
        # NOTE: The creators of the Pascal VOC dataset started counting at 1,
        # and thus the indices have to be corrected.
        xmin = (float(bbox.find('xmin').text) - 1.) / width
        xmax = (float(bbox.find('xmax').text) - 1.) / width
        ymin = (float(bbox.find('ymin').text) - 1.) / height
        ymax = (float(bbox.find('ymax').text) - 1.) / height
        annotations.append((name, xmin, ymin, xmax, ymax, truncated, difficult))

    return annotations


def to_numpy_image(image, size, normalize=True):
    """
    A utility function that converts a Tensor in the range [0., 1.] to a
    resized ndarray in the range [0, 255].
    Parameters
    ----------
    image : Tensor
        A Tensor representing the image.
    size : tuple of int
        The size (w, h) to which the image should be resized.
    normalize : bool
        A flag which indicates whether the image was orignially normalized,
        which means that it should be de-normalized when converting to an
        array.
    Returns
    -------
    image : ndarray
        A ndarray representation of the image.
    """
    image = np.array(image.permute(1, 2, 0).cpu().numpy())
    if normalize:
        image *= VGG_STD
        image += VGG_MEAN
    image *= 255.
    image = image.astype(dtype=np.uint8)
    image = cv2.resize(image, dsize=size, interpolation=cv2.INTER_CUBIC)

    return image


def add_bbox_to_image(image, bbox, confidence, cls):
    """
    A utility function that adds a bounding box with labels to an image in-place.

    Parameters
    ----------
    image : ndarray
        An ndarray containing the image.
    bbox : array_like
        An array (x1, y1, x2, y2) containing the coordinates of the upper-
        left and bottom-right corners of the bounding box to be added to
        the image.
    confidence : float
        A value representing the confidence of an object within the bounding
        box. This value will be displayed as part of the label.
    cls : str
        The class to which the object in the bounding box belongs. This
        value will be displayed as part of the label.
    """
    text = '{} {:.2f}'.format(cls, confidence)
    height, width = image.shape[:2]
    xmin, ymin, xmax, ymax = bbox
    xmin *= width
    xmax *= width
    ymin *= height
    ymax *= height
    # Draw a bounding box.
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
    assert boxes_a.shape[-1] == 4
    assert boxes_b.shape[-1] == 4
    assert isinstance(boxes_a, torch.Tensor)
    assert isinstance(boxes_b, torch.Tensor)

    tl = torch.max(boxes_a[:, None, :2], boxes_b[:, :2])
    br = torch.min(boxes_a[:, None, 2:], boxes_b[:, 2:])

    area_a = torch.prod(boxes_a[:, 2:] - boxes_a[:, :2], 1)
    area_b = torch.prod(boxes_b[:, 2:] - boxes_b[:, :2], 1)

    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en

    area_a = torch.clamp(area_a, min=0)

    ious = area_i / (area_a[:, None] + area_b - area_i)

    return ious


def index_dict_list(dictionary, index):

    try:
        return {k: v[index] for k,v in dictionary.items() if isinstance(v, (list, torch.Tensor))}
    except IndexError:
        return None


def random_choice(x, size, replace=False):

    idx = torch.multinomial(torch.ones(x.numel()), size, replacement=replace)

    return x[idx].squeeze()


def xyxy2xywh(xyxy):
    """
    Utility function that converts bounding boxes that are in the form
    (x1, y1, x2, y2) to (x_c, y_c, w, h).
    Parameters
    ----------
    xyxy : Tensor
        A tensor whose shape is :math:`(N, 4)` where the elements in the
        0th and 1st column represent the x and y coordinates of the top left
        corner of the bounding box and the 2nd and 3rd column represent the
        x and y coordinates of the bottom right corner of the bounding box.
    Returns
    -------
    xywh : Tensor
        A tensor whose shape is :math:`(N, 4)` where the elements in the
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
    Converts bounding boxes that are in the form
    (x_c, y_c, w, h) to (x1, y1, x2, y2).
    Parameters
    ----------
    xywh : Tensor
        A tensor whose shape is :math:`(N, 4)` where the elements in the
        0th and 1st column represent the x and y coordinates of the center
        of the bounding box and the 2nd and 3rd column represent the
        width and height of the bounding box.

    Returns
    -------
    xyxy : Tensor
        A tensor whose shape is :math:`(N, 4)` where the elements in the
        0th and 1st column represent the x and y coordinates of the top left
        corner of the bounding box and the 2nd and 3rd column represent the
        x and y coordinates of the bottom right corner of the bounding box.
    """
    xyxy = torch.zeros_like(xywh)
    half = xywh[:, 2:] / 2.
    xyxy[:, :2] = xywh[:, :2] - half
    xyxy[:, 2:] = xywh[:, :2] + half

    return xyxy


def parameterize_bboxes(bboxes, anchors):

    assert bboxes.shape == anchors.shape
    assert bboxes.shape[-1] == 4

    bboxes_xywh = xyxy2xywh(bboxes)
    anchors_xywh = xyxy2xywh(anchors)

    bboxes_xywh[:, :2] = (bboxes_xywh[:, :2] - anchors_xywh[:, :2]) / anchors_xywh[:, 2:]
    bboxes_xywh[:, 2:] = torch.log(bboxes_xywh[:, 2:] / anchors_xywh[:, 2:])

    return bboxes_xywh


def deparameterize_bboxes(reg, anchors):

    assert reg.shape == anchors.shape
    assert anchors.shape[-1] == 4

    anchors_xywh = xyxy2xywh(anchors)
    bboxes_xywh = torch.zeros_like(anchors)

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


# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def non_maximum_suppression(boxes, scores, overlap=0.5, top_k=101):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The predictions for the image, Shape: [predictions, 4].
        scores: (tensor) The class scores for the image, Shape:[num_priors].
        overlap: (float) The overlap threshold for suppressing unnecessary boxes.
        top_k: (int) The maximum number of predictions to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """
    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w * h
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        iou = inter / union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[iou.le(overlap)]

    return keep[:count].clone()
