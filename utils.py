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
        should be normalised to the height of the image.
    left : float
        The x-coordinate of the top-left corner of the predicted bounding box. The value
        should be normalised to the width of the image.
    bottom : float
        The y-coordinate of the bottom-right corner of the predicted bounding box. The value
        should be normalised to the height of the image.
    right : float
        The x-coordinate of the bottom-right corner of the predicted bounding box. The value
        should be normalised to the width of the image.
    confidence : float
        A confidence value attached to the prediction, which should be generated by the
        detector.  The value does not have to be normalised, but a greater value corresponds
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
        top = np.maximum(top, 1.)
        left = np.maximum(left, 1.)
        bottom = np.maximum(bottom, 1.)
        right = np.maximum(right, 1.)
        prediction = [image_id, confidence, np.round(left), np.round(top), np.round(right), np.round(bottom), '\n']
        prediction = map(to_repr, prediction)
        prediction = ' '.join(prediction)
        f.write(prediction)
