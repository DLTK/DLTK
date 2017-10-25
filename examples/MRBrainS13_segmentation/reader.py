import SimpleITK as sitk
import tensorflow as tf
import os

from dltk.io.augmentation import *
from dltk.io.preprocessing import *

def receiver(file_references, mode, params=None):
    """Summary
    
    Args:
        file_references (TYPE): Description
        mode (TYPE): Description
        params (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    
    def _augment(img, lbl):
        
        img = add_gaussian_noise(img, sigma=0.1)
        [img, lbl] = flip([img, lbl], axis=1)
        
        return img, lbl

    n_examples = params['n_examples']
    example_size = params['example_size']

    i = 0
    while True:

        # NEEDS TO BE FIXED!
        img_fn = file_references[i][1]
        i += 1
        if i == len(file_references):
            i = 0

        t1 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(str(img_fn), 'T1.nii')))
        t1_ir = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(str(img_fn), 'T1_IR.nii')))
        t2_fl = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(str(img_fn), 'T2_FLAIR.nii')))

        # Normalise volume images
        t1 = whitening(t1)
        t1_ir = whitening(t1_ir)
        t2_fl = whitening(t2_fl)

        # Create a 4D multi-sequence image (i.e. [channels, x, y, z])
        images = np.asarray([t1, t1_ir, t2_fl]).astype(np.float32)

        # Transpose to [batch, x, y, z, channel] as required input by the network
        images = np.transpose(images, (1, 2, 3, 0))

        if mode == tf.estimator.ModeKeys.PREDICT:
            yield {'features': {'x': images}, 'labels': None}

        lbl = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(str(img_fn), 'LabelsForTraining.nii'))).astype(
            np.int32)

        # Augment if used in training mode
        if mode == tf.estimator.ModeKeys.TRAIN:
            images, lbl = _augment(images, lbl)
        
        # Check if the reader is supposed to return training examples or full images
        if params['extract_examples']:
            images, lbl = extract_class_balanced_example_array(images, lbl, example_size=example_size,
                                                               n_examples=n_examples, classes=9)
            for e in range(n_examples):
                yield {'features': {'x': images[e].astype(np.float32)}, 'labels': {'y': lbl[e].astype(np.int32)}}
        else:
            yield {'features': {'x': images}, 'labels': {'y': lbl}}

    return


def save_fn(file_reference, data, output_path):
    """Summary
    
    Args:
        file_references (TYPE): Description
        data (TYPE): Description
        output_path (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    lbl = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(str(file_reference), 'LabelsForTraining.nii')))

    new_sitk = sitk.GetImageFromArray(data)

    new_sitk.CopyInformation(lbl)

    sitk.WriteImage(new_sitk, output_path)