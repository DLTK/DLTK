from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def flip(imglist, axis=1):
    """ Randomly flip spatial dimensions

    Parameters
    ----------
    imglist : np.ndarray or list or tuple
        image(s) to be flipped
    axis : int
        axis along which to flip the images

    Returns
    -------
    np.ndarray or list or tuple
        same as imglist but randomly flipped along axis

    """
    was_singular = False
    if isinstance(imglist, np.ndarray):
        imglist = [imglist]
        was_singular = True

    do_flip = np.random.random(1)
    if do_flip > 0.5:
        for i in range(len(imglist)):
            imglist[i] = np.flip(imglist[i], axis=axis)
    if was_singular:
        return imglist[0]
    return imglist


def gaussian_offset(img, sigma=0.1):
    """ Add Gaussian offset to an image

    Adds the offset to each channel independently

    Parameters
    ----------
    img : np.ndarray
        image to add noise to
    sigma : float
        stddev for normal distribution to generate noise from

    Returns
    -------
    np.ndarray
        same as image but with added offset to each channel

    """

    offsets = np.random.normal(0, sigma, ([1] * (img.ndim - 1) + [img.shape[-1]]))
    img += offsets
    return img


def gaussian_noise(img, sigma=0.05):
    """ Add Gaussian noise to an image

    Parameters
    ----------
    img : np.ndarray
        image to add noise to
    sigma : float
        stddev for normal distribution to generate noise from

    Returns
    -------
    np.ndarray
        same as image but with added noise
    """

    img += np.random.normal(0, sigma, img.shape)
    return img


def extract_class_balanced_example_array(image, label, example_size=[1, 64, 64], n_examples=1, classes=2):
    """
        Extract training examples from an image (and corresponding label) subject to class balancing.
        Returns an image example array and the corresponding label array.

        Parameters
        ----------
        image: np.ndarray
            image to extract class-balanced patches from
        label: np.ndarray
            labels to use for balancing the classes
        example_size: list or tuple
            shape of the patches to extract
        n_examples: int
            number of patches to extract in total
        classes : int or list or tuple
            number of classes or list of classes to extract

        Returns
        -------
        ex_imgs, ex_lbls
            class-balanced patches extracted from bigger images with shape [batch, example_size..., image_channels]
    """
    assert(image.shape[:-1] == label.shape), 'Image and label shape must match'
    assert(image.ndim - 1 == len(example_size), 'Example size doesnt fit image size')
    assert(all([i_s > e_s for i_s, e_s in zip(image.shape, example_size)]),
           'Image must be bigger than example shape')
    rank = len(example_size)

    if isinstance(classes, int):
        classes = tuple(range(classes))
    n_classes = len(classes)

    n_ex_per_class = int(np.round(n_examples / n_classes))

    # compute an example radius as we are extracting centered around locations
    ex_rad = np.array(list(zip(np.floor(np.array(example_size) / 2.0), np.ceil(np.array(example_size) / 2.0))),
                      dtype=np.int)

    ex_imgs = []
    ex_lbls = []
    for c in classes:
        # get valid, random center locations belonging to that class
        idx = np.argwhere(label == c)

        # extract random locations
        r_idx_idx = np.random.choice(len(idx), size=min(n_ex_per_class, len(idx)), replace=False).astype(int)
        r_idx = idx[r_idx_idx]

        # add a random shift them to avoid learning a centre bias - IS THIS REALLY TRUE?
        r_shift = np.array([list(a) for a in zip(
                    *[np.random.randint(-ex_rad[i][0], ex_rad[i][1], size=min(n_ex_per_class, len(idx))) for i in range(rank)]
                  )])

        r_idx += r_shift

        # shift them to valid locations if necessary
        r_idx = np.array([np.array([max(min(r[dim], image.shape[dim] - ex_rad[dim][1]),
                                        ex_rad[dim][0]) for dim in range(rank)]) for r in r_idx])

        for i in range(len(r_idx)):
            # extract class-balanced examples from the original image
            slicer = [slice(r_idx[i][dim] - ex_rad[dim][0], r_idx[i][dim] + ex_rad[dim][1]) for dim in range(rank)]
            ex_img = image[slicer][np.newaxis, :]

            ex_lbl = label[slicer][np.newaxis, :]

            # concatenate and return the examples
            ex_imgs = np.concatenate((ex_imgs, ex_img), axis=0) if (len(ex_imgs) != 0) else ex_img
            ex_lbls = np.concatenate((ex_lbls, ex_lbl), axis=0) if (len(ex_lbls) != 0) else ex_lbl

    return ex_imgs, ex_lbls


def extract_random_example_array(image_list, example_size=[1, 64, 64], n_examples=1):
    """
        Randomly extract training examples from image (and corresponding label).
        Returns an image example array and the corresponding label array.

        Parameters
        ----------
        image_list: np.ndarray or list or tuple
            image(s) to extract random patches from
        example_size: list or tuple
            shape of the patches to extract
        n_examples: int
            number of patches to extract in total

        Returns
        -------
        examples
            random patches extracted from bigger images with same type as image_list with of shape
            [batch, example_size..., image_channels]
    """
    assert n_examples > 0

    was_singular = False
    if isinstance(image_list, np.ndarray):
        image_list = [image_list]
        was_singular = True

    assert (all([i_s > e_s for i_s, e_s in zip(image_list[0].shape, example_size)]),
            'Image must be bigger than example shape')
    assert ((image_list[0].ndim - 1 == len(example_size) or image_list[0].ndim == len(example_size)),
            'Example size doesnt fit image size')

    for i in image_list:
        if len(image_list) > 1:
            assert ((i.ndim - 1 == image_list[0].ndim or i.ndim == image_list[0].ndim),
                    'Example size doesnt fit image size')
            assert (all([i0_s == i_s for i0_s, i_s in zip(image_list[0].shape, i.shape)]),
                    'Image shapes must match')

    rank = len(example_size)

    # extract random examples from image and label
    valid_loc_range = [image_list[0].shape[i] - example_size[i] for i in range(rank)]

    rnd_loc = [np.random.randint(valid_loc_range[dim], size=n_examples)
                if valid_loc_range[dim] > 0 else np.zeros(n_examples, dtype=int) for dim in range(rank)]

    examples = [[]] * len(image_list)
    for i in range(n_examples):
        slicer = [slice(rnd_loc[dim][i], rnd_loc[dim][i] + example_size[dim]) for dim in range(rank)]

        for j in range(len(image_list)):
            ex_img = image_list[j][slicer][np.newaxis]
            # concatenate and return the examples
            examples[j] = np.concatenate((examples[j], ex_img), axis=0) if (len(examples[j]) != 0) else ex_img

    if was_singular:
        return examples[0]
    return examples