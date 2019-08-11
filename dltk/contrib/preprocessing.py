from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import cv2


def whitening(image):
    """Whitening. Normalises image to zero mean and unit variance."""

    image = image.astype(np.float32)

    mean = np.mean(image)
    std = np.std(image)

    if std > 0:
        ret = (image - mean) / std
    else:
        ret = image * 0.
    return ret


def erode_image(image, kernel_dims=(5, 5), num_iterations=1):
    """ Eroding image, dims boundaries of background object. """
    if kernel_dims[0] != kernel_dims[1]:
        raise ValueError('kernel dimensions do not match')
    kernel = np.ones(kernel_dims, np.uint8)
    erosion = cv2.erode(image, kernel, iterations=num_iterations)
    return erosion


def dilate_image(image, kernel_dims=(5, 5), num_iterations=1):
    """ Dilating image, enlarges boundaries of background object. """
    if kernel_dims[0] != kernel_dims[1]:
        raise ValueError('kernel dimensions do not match')
    kernel = np.ones(kernel_dims, np.uint8)
    dilation = cv2.dilate(image, kernel, iterations=num_iterations)
    return dilation


def open_image(image, kernel_dims=(5, 5)):
    """ Erosion followed by dilation. """
    if kernel_dims[0] != kernel_dims[1]:
        raise ValueError('kernel dimensions do not match')
    kernel = np.ones(kernel_dims, np.uint8)
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return opening


def close_image(image, kernel_dims=(5, 5)):
    """ Dilation followed by erosion. """
    if kernel_dims[0] != kernel_dims[1]:
        raise ValueError('kernel dimensions do not match')
    kernel = np.ones(kernel_dims, np.uint8)
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return closing


def morph_gradient_image(image, kernel_dims=(5, 5)):
    """ Difference between opening and closing. """
    if kernel_dims[0] != kernel_dims[1]:
        raise ValueError('kernel dimensions do not match')
    kernel = np.ones(kernel_dims, np.uint8)
    gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    return gradient


def top_hat_image(image, kernel_dims=(5, 5)):
    """ Difference between input image and opening of the same image. """
    if kernel_dims[0] != kernel_dims[1]:
        raise ValueError('kernel dimensions do not match')
    kernel = np.ones(kernel_dims, np.uint8)
    top_hat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    return top_hat


def black_hat_image(image, kernel_dims=(5, 5)):
    """ Difference between closing of the input image and the image. """
    if kernel_dims[0] != kernel_dims[1]:
        raise ValueError('kernel dimensions do not match')
    kernel = np.ones(kernel_dims, np.uint8)
    black_hat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    return black_hat

    
def normalise_zero_one(image):
    """Image normalisation. Normalises image to fit [0, 1] range."""

    image = image.astype(np.float32)

    minimum = np.min(image)
    maximum = np.max(image)

    if maximum > minimum:
        ret = (image - minimum) / (maximum - minimum)
    else:
        ret = image * 0.
    return ret


def normalise_one_one(image):
    """Image normalisation. Normalises image to fit [-1, 1] range."""

    ret = normalise_zero_one(image)
    ret *= 2.
    ret -= 1.
    return ret


def resize_image_with_crop_or_pad(image, img_size=(64, 64, 64), **kwargs):
    """Image resizing. Resizes image by cropping or padding dimension
     to fit specified size.

    Args:
        image (np.ndarray): image to be resized
        img_size (list or tuple): new image size
        kwargs (): additional arguments to be passed to np.pad

    Returns:
        np.ndarray: resized image
    """

    assert isinstance(image, (np.ndarray, np.generic))
    assert (image.ndim - 1 == len(img_size) or image.ndim == len(img_size)), \
        'Example size doesnt fit image size'

    # Get the image dimensionality
    rank = len(img_size)

    # Create placeholders for the new shape
    from_indices = [[0, image.shape[dim]] for dim in range(rank)]
    to_padding = [[0, 0] for dim in range(rank)]

    slicer = [slice(None)] * rank

    # For each dimensions find whether it is supposed to be cropped or padded
    for i in range(rank):
        if image.shape[i] < img_size[i]:
            to_padding[i][0] = (img_size[i] - image.shape[i]) // 2
            to_padding[i][1] = img_size[i] - image.shape[i] - to_padding[i][0]
        else:
            from_indices[i][0] = int(np.floor((image.shape[i] - img_size[i]) / 2.))
            from_indices[i][1] = from_indices[i][0] + img_size[i]

        # Create slicer object to crop or leave each dimension
        slicer[i] = slice(from_indices[i][0], from_indices[i][1])

    # Pad the cropped image to extend the missing dimension
    return np.pad(image[slicer], to_padding, **kwargs)
