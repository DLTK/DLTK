from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


def flip(imagelist, axis=1):
    """Randomly flip spatial dimensions

    Args:
        imagelist (np.ndarray or list or tuple): image(s) to be flipped
        axis (int): axis along which to flip the images

    Returns:
        np.ndarray or list or tuple: same as imagelist but randomly flipped
            along axis
    """

    # Check if a single image or a list of images has been passed
    was_singular = False
    if isinstance(imagelist, np.ndarray):
        imagelist = [imagelist]
        was_singular = True

    # With a probility of 0.5 flip the image(s) across `axis`
    do_flip = np.random.random(1)
    if do_flip > 0.5:
        for i in range(len(imagelist)):
            imagelist[i] = np.flip(imagelist[i], axis=axis)
    if was_singular:
        return imagelist[0]
    return imagelist


def add_gaussian_offset(image, sigma=0.1):
    """
    Add Gaussian offset to an image. Adds the offset to each channel
    independently.

    Args:
        image (np.ndarray): image to add noise to
        sigma (float): stddev of the Gaussian distribution to generate noise
            from

    Returns:
        np.ndarray: same as image but with added offset to each channel
    """

    offsets = np.random.normal(0, sigma, ([1] * (image.ndim - 1) + [image.shape[-1]]))
    image += offsets
    return image


def add_gaussian_noise(image, sigma=0.05):
    """
    Add Gaussian noise to an image

    Args:
        image (np.ndarray): image to add noise to
        sigma (float): stddev of the Gaussian distribution to generate noise
            from

    Returns:
        np.ndarray: same as image but with added offset to each channel
    """

    image += np.random.normal(0, sigma, image.shape)
    return image


def elastic_transform(image, alpha, sigma):
    """
    Elastic deformation of images as described in [1].

    [1] Simard, Steinkraus and Platt, "Best Practices for Convolutional
        Neural Networks applied to Visual Document Analysis", in Proc. of the
        International Conference on Document Analysis and Recognition, 2003.

    Based on gist https://gist.github.com/erniejunior/601cdf56d2b424757de5

    Args:
        image (np.ndarray): image to be deformed
        alpha (list): scale of transformation for each dimension, where larger
            values have more deformation
        sigma (list): Gaussian window of deformation for each dimension, where
            smaller values have more localised deformation

    Returns:
        np.ndarray: deformed image
    """

    assert len(alpha) == len(sigma), \
        "Dimensions of alpha and sigma are different"

    channelbool = image.ndim - len(alpha)
    out = np.zeros((len(alpha) + channelbool, ) + image.shape)

    # Generate a Gaussian filter, leaving channel dimensions zeroes
    for jj in range(len(alpha)):
        array = (np.random.rand(*image.shape) * 2 - 1)
        out[jj] = gaussian_filter(array, sigma[jj],
                                  mode="constant", cval=0) * alpha[jj]

    # Map mask to indices
    shapes = list(map(lambda x: slice(0, x, None), image.shape))
    grid = np.broadcast_arrays(*np.ogrid[shapes])
    indices = list(map((lambda x: np.reshape(x, (-1, 1))), grid + np.array(out)))

    # Transform image based on masked indices
    transformed_image = map_coordinates(image, indices, order=0,
                                        mode='reflect').reshape(image.shape)

    return transformed_image


def extract_class_balanced_example_array(image,
                                         label,
                                         example_size=[1, 64, 64],
                                         n_examples=1,
                                         classes=2,
                                         class_weights=None):
    """Extract training examples from an image (and corresponding label) subject
        to class balancing. Returns an image example array and the
        corresponding label array.

    Args:
        image (np.ndarray): image to extract class-balanced patches from
        label (np.ndarray): labels to use for balancing the classes
        example_size (list or tuple): shape of the patches to extract
        n_examples (int): number of patches to extract in total
        classes (int or list or tuple): number of classes or list of classes
            to extract

    Returns:
        np.ndarray, np.ndarray: class-balanced patches extracted from full
            images with the shape [batch, example_size..., image_channels]
    """
    assert image.shape[:-1] == label.shape, 'Image and label shape must match'
    assert image.ndim - 1 == len(example_size), \
        'Example size doesnt fit image size'
    assert all([i_s >= e_s for i_s, e_s in zip(image.shape, example_size)]), \
        'Image must be larger than example shape'
    rank = len(example_size)

    if isinstance(classes, int):
        classes = tuple(range(classes))
    n_classes = len(classes)

    assert n_examples >= n_classes, \
        'n_examples need to be greater than n_classes'

    if class_weights is None:
        n_ex_per_class = np.ones(n_classes).astype(int) * int(np.round(n_examples / n_classes))
    else:
        assert len(class_weights) == n_classes, \
            'Class_weights must match number of classes'
        class_weights = np.array(class_weights)
        n_ex_per_class = np.round((class_weights / class_weights.sum()) * n_examples).astype(int)

    # Compute an example radius to define the region to extract around a
    # center location
    ex_rad = np.array(list(zip(np.floor(np.array(example_size) / 2.0),
                               np.ceil(np.array(example_size) / 2.0))),
                      dtype=np.int)

    class_ex_images = []
    class_ex_lbls = []
    min_ratio = 1.
    for c_idx, c in enumerate(classes):
        # Get valid, random center locations belonging to that class
        idx = np.argwhere(label == c)

        ex_images = []
        ex_lbls = []

        if len(idx) == 0 or n_ex_per_class[c_idx] == 0:
            class_ex_images.append([])
            class_ex_lbls.append([])
            continue

        # Extract random locations
        r_idx_idx = np.random.choice(len(idx),
                                     size=min(n_ex_per_class[c_idx], len(idx)),
                                     replace=False).astype(int)
        r_idx = idx[r_idx_idx]

        # Shift the random to valid locations if necessary
        r_idx = np.array(
            [np.array([max(min(r[dim], image.shape[dim] - ex_rad[dim][1]),
                           ex_rad[dim][0]) for dim in range(rank)])
             for r in r_idx])

        for i in range(len(r_idx)):
            # Extract class-balanced examples from the original image
            slicer = [slice(r_idx[i][dim] -
                            ex_rad[dim][0], r_idx[i][dim] +
                            ex_rad[dim][1]) for dim in range(rank)]

            ex_image = image[slicer][np.newaxis, :]

            ex_lbl = label[slicer][np.newaxis, :]

            # Concatenate them and return the examples
            ex_images = np.concatenate((ex_images, ex_image), axis=0) \
                if (len(ex_images) != 0) else ex_image
            ex_lbls = np.concatenate((ex_lbls, ex_lbl), axis=0) \
                if (len(ex_lbls) != 0) else ex_lbl

        class_ex_images.append(ex_images)
        class_ex_lbls.append(ex_lbls)

        ratio = n_ex_per_class[c_idx] / len(ex_images)
        min_ratio = ratio if ratio < min_ratio else min_ratio

    indices = np.floor(n_ex_per_class * min_ratio).astype(int)

    ex_images = np.concatenate([cimage[:idxs] for cimage, idxs in zip(class_ex_images, indices)
                                if len(cimage) > 0], axis=0)
    ex_lbls = np.concatenate([clbl[:idxs] for clbl, idxs in zip(class_ex_lbls, indices)
                              if len(clbl) > 0], axis=0)

    return ex_images, ex_lbls


def extract_random_example_array(image_list,
                                 example_size=[1, 64, 64],
                                 n_examples=1):
    """Randomly extract training examples from image (and a corresponding label).
        Returns an image example array and the corresponding label array.

    Args:
        image_list (np.ndarray or list or tuple): image(s) to extract random
            patches from
        example_size (list or tuple): shape of the patches to extract
        n_examples (int): number of patches to extract in total

    Returns:
        np.ndarray, np.ndarray: class-balanced patches extracted from full
        images with the shape [batch, example_size..., image_channels]
    """

    assert n_examples > 0

    was_singular = False
    if isinstance(image_list, np.ndarray):
        image_list = [image_list]
        was_singular = True

    assert all([i_s >= e_s for i_s, e_s in zip(image_list[0].shape, example_size)]), \
        'Image must be bigger than example shape'
    assert (image_list[0].ndim - 1 == len(example_size) or
            image_list[0].ndim == len(example_size)), \
        'Example size doesnt fit image size'

    for i in image_list:
        if len(image_list) > 1:
            assert (i.ndim - 1 == image_list[0].ndim or
                    i.ndim == image_list[0].ndim or
                    i.ndim + 1 == image_list[0].ndim), \
                'Example size doesnt fit image size'

            assert all([i0_s == i_s for i0_s, i_s in zip(image_list[0].shape, i.shape)]), \
                'Image shapes must match'

    rank = len(example_size)

    # Extract random examples from image and label
    valid_loc_range = [image_list[0].shape[i] - example_size[i] for i in range(rank)]

    rnd_loc = [np.random.randint(valid_loc_range[dim], size=n_examples)
               if valid_loc_range[dim] > 0
               else np.zeros(n_examples, dtype=int) for dim in range(rank)]

    examples = [[]] * len(image_list)
    for i in range(n_examples):
        slicer = [slice(rnd_loc[dim][i], rnd_loc[dim][i] + example_size[dim])
                  for dim in range(rank)]

        for j in range(len(image_list)):
            ex_image = image_list[j][slicer][np.newaxis]
            # Concatenate and return the examples
            examples[j] = np.concatenate((examples[j], ex_image), axis=0) \
                if (len(examples[j]) != 0) else ex_image

    if was_singular:
        return examples[0]
    return examples
