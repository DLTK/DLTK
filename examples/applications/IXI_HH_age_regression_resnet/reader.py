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
        
        #images = add_gaussian_noise(images, sigma=0.1)
        images = flip(images, axis=2)
        
        return images

    i = 0
    while True:

        subject_id = file_references[i][0]
        i += 1
        if i == len(file_references):
            i = 0
            
        data_path = '../../../data/IXI_HH/2mm'
        
        t1_fn = os.path.join(data_path, '{}/T1_2mm.nii.gz'.format(subject_id))
        #t2_fn = os.path.join(data_path, '{}/T2_1mm.nii.gz'.format(subject_id))
        #pd_fn = os.path.join(data_path, '{}/PD_1mm.nii.gz'.format(subject_id))
        
        t1 = sitk.GetArrayFromImage(sitk.ReadImage(t1_fn))
        #t2 = sitk.GetArrayFromImage(sitk.ReadImage(t2_fn))
        #pd = sitk.GetArrayFromImage(sitk.ReadImage(pd_fn))

        # Normalise volume images
        t1 = whitening(t1)
        #t2 = whitening(t2)
        #pd = whitening(pd)

        # Create a 4D multi-sequence image (i.e. [channels, x, y, z])
        #images = np.asarray([t1, t2, pd]).astype(np.float32)
        
        # Transpose to [batch, x, y, z, channel] as required input by the network
        #images = np.transpose(images, (1, 2, 3, 0))
        
        images = np.expand_dims(t1, axis=-1).astype(np.float32)

        
        if mode == tf.estimator.ModeKeys.PREDICT:
            yield {'features': {'x': images}}

        # Parse the regression targets from the file_references
        age = np.float(file_references[i][11])
        y = np.expand_dims(age, axis=-1).astype(np.float32)
            
        # Augment if used in training mode
        if mode == tf.estimator.ModeKeys.TRAIN:
            images = _augment(images)
        
        # Check if the reader is supposed to return training examples or full images
        if params['extract_examples']:
            images = extract_random_example_array(images, example_size=params['example_size'], n_examples=params['n_examples'])
            
            for e in range(params['n_examples']):
                yield {'features': {'x': images[e].astype(np.float32)}, 'labels': {'y': y.astype(np.float32)}}
                       
        else:
            yield {'features': {'x': images}, 'labels': {'y': y.astype(np.float32)}}

    return