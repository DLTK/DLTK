import SimpleITK as sitk
import tensorflow as tf
import os
import glob

from dltk.io.augmentation import *
from dltk.io.preprocessing import *
import scipy

def read_fn(file_references, mode, params=None):
    """Summary
    
    Args:
        file_references (TYPE): Description
        mode (TYPE): Description
        params (TYPE): Description
    
    Returns:
        TYPE: Description
    """

    for f in file_references:
        subject_id = f[0]
            
        data_path = '../../../data/IXI_HH/1mm'
        
        t1_fn = os.path.join(data_path, '{}/T1_1mm.nii.gz'.format(subject_id))
        
        t1 = sitk.GetArrayFromImage(sitk.ReadImage(t1_fn))

        # Normalise volume images
        t1 = t1[..., np.newaxis]
        
        # restrict to slices around center slice
        t1 = t1[len(t1) // 2 - 5:len(t1) // 2 + 5]
        
        t1 = normalise_zero_one(t1)
        
        images = t1

        noise = np.random.normal(size=(1, 1, 1, 100))

        if mode == tf.estimator.ModeKeys.PREDICT:
            yield {'labels': images, 'features': {'noise': noise}}
        
        # Check if the reader is supposed to return training examples or full images
        if params['extract_examples']:
            images = extract_random_example_array(images, example_size=params['example_size'],
                                                  n_examples=params['n_examples'])
            for e in range(params['n_examples']):
                
                yield {'labels': scipy.ndimage.zoom(images[e], (1, 64./224., 64./224., 1)).astype(np.float32),
                       'features': {'noise': noise}}
        else:
            yield {'labels': images, 'features': {'noise': noise}}

    return