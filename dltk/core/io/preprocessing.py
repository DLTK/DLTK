from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np


def whitening(image):
    """Whitening

    Normalises image to zero mean and unit variance

    Parameters
    ----------
    image : np.ndarray
        image to be whitened

    Returns
    -------
    np.ndarray
        whitened image

    """
    ret = (image - np.mean(image)) / np.std(image)
    return ret


def normalise_zero_one(image):
    """Image normalisation

    Normalises image to fit [0, 1] range

    Parameters
    ----------
    image : np.ndarray
        image to be normalised

    Returns
    -------
    np.ndarray
        normalised image

    """
    image = image.astype(np.float32)
    ret = (image - np.min(image))
    ret /= np.max(image)

    return ret


def normalise_one_one(image):
    """Image normalisation

    Normalises image to fit [-1, 1] range

    Parameters
    ----------
    image : np.ndarray
        image to be normalised

    Returns
    -------
    np.ndarray
        normalised image

    """
    ret = normalise_zero_one(image)
    ret *= 2.
    ret -= 1.

    return ret


def resize_image_with_crop_or_pad(image, img_size=[64,64,64], **kwargs):
    """Image resizing

    Resizes image by cropping or padding dimension to fit specified size.

    Parameters
    ----------
    image : np.ndarray
        image to be resized
    img_size : list or tuple
        new image size
    kwargs
        additional arguments to be passed to np.pad

    Returns
    -------
    np.ndarray
        resized image

    """

    assert isinstance(image, (np.ndarray, np.generic))
    assert ((image.ndim - 1 == len(img_size) or image.ndim == len(img_size)),
            'Example size doesnt fit image size')

    # find image dimensionality
    rank = len(img_size)

    # create placeholders for new shape
    from_indices = [[0, image.shape[dim]] for dim in range(rank)]
    to_padding = [[0, img_size[dim]] for dim in range(rank)]

    slicer = [slice(None)] * rank

    for i in range(rank):
        # for each dimensions find whether it is supposed to be cropped or padded
        if image.shape[i + 1] <= img_size[i]:
            to_padding[i][0] = (img_size[i] - image.shape[i]) // 2
            to_padding[i][1] = img_size[i] - image.shape[i] - to_padding[i][0]
        else:
            from_indices[i][0] = int(np.floor((image.shape[i] - img_size[i]) / 2.))
            from_indices[i][1] = from_indices[i][0] + img_size[i]

        # create slicer object to crop or leave each dimension
        slicer[i] = slice(from_indices[i][0], from_indices[i][1])

    # pad the cropped image to extend the missing dimension
    return np.pad(image[slicer], to_padding, **kwargs)