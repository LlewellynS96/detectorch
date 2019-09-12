import os
import cv2
import numpy as np
import xml.etree.ElementTree as Et
import torch


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


def to_numpy_image(image, size):
    """
    A utility function that converts a Tensor in the range [0., 1.] to a
    resized ndarray in the range [0, 255].
    Parameters
    ----------
    image : Tensor
        A Tensor representing the image.
    size : tuple of int
        The size (w, h) to which the image should be resized.

    Returns
    -------
    image : ndarray
        A ndarray representation of the image.
    """
    image = image.permute(1, 2, 0).cpu().numpy()
    image *= 255.
    image = image.astype(dtype=np.uint8)
    image = cv2.resize(image, dsize=size, interpolation=cv2.INTER_CUBIC)
    image = np.array(image)

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


def index_dict_list(dictionary, index):

    return {k: v[index] for k,v in dictionary.items() if isinstance(v, (list, torch.Tensor))}
