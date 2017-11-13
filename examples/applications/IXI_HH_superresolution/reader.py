import SimpleITK as sitk
import tensorflow as tf
import os
import glob

from dltk.io.augmentation import *
from dltk.io.preprocessing import *

def read_fn(file_references, mode, params=None):
    """Summary
    
    Args:
        file_references (TYPE): Description
        mode (TYPE): Description
        params (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    
    def _augment(images):
        return flip(images, axis=2)

    i = 0
    while True:

        subject_id = file_references[i][0]
        i += 1
        if i == len(file_references):
            i = 0
            
        data_path = '../../../data/IXI_HH/1mm'
        
        t1_fn = os.path.join(data_path, '{}/T1_1mm.nii.gz'.format(subject_id))
        
        t1 = sitk.GetArrayFromImage(sitk.ReadImage(t1_fn))

        # Normalise volume images
        t1 = whitening(t1)

        # Create a 4D image (i.e. [x, y, z, channels])
        images = np.expand_dims(t1, axis=-1).astype(np.float32)

        if mode == tf.estimator.ModeKeys.PREDICT:
            yield {'features': {'x': images}}

        # Augment if used in training mode
        if mode == tf.estimator.ModeKeys.TRAIN:
            images = _augment(images)
        
        # Check if the reader is supposed to return training examples or full images
        if params['extract_examples']:
            images = extract_random_example_array(images, example_size=params['example_size'], n_examples=params['n_examples'])
            
            for e in range(params['n_examples']):
                yield {'features': {'x': images[e].astype(np.float32)}}
        else:
            yield {'features': {'x': images}}

    return