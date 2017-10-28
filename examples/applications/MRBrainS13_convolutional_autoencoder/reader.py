import SimpleITK as sitk
import tensorflow as tf
import os
import glob

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
    
    def _augment(img):
        
        img = add_gaussian_noise(img, sigma=0.1)
        img = flip(img, axis=1)
        
        return img

    i = 0
    while True:

        subject_id = file_references[i][0]
        i += 1
        if i == len(file_references):
            i = 0
            
        data_path = '../../../data/IXI_HH/'
        
        t1_fn = glob.glob(data_path + 't1/{}*.nii.gz'.format(subject_id))[0]
        t2_fn = glob.glob(data_path + 't2/{}*.nii.gz'.format(subject_id))[0]
        pd_fn = glob.glob(data_path + 'pd/{}*.nii.gz'.format(subject_id))[0]
        mra_fn = glob.glob(data_path + 'mra/{}*.nii.gz'.format(subject_id))[0]
        
        

        t1 = sitk.GetArrayFromImage(sitk.ReadImage(t1_fn))
        t2 = sitk.GetArrayFromImage(sitk.ReadImage(t2_fn))
        pd = sitk.GetArrayFromImage(sitk.ReadImage(pd_fn))
        mra = sitk.GetArrayFromImage(sitk.ReadImage(mra_fn))
        
        print(t1.shape, t2.shape, pd.shape, mra.shape)

        # Normalise volume images
        t1 = whitening(t1)
        t2 = whitening(t2)
        pd = whitening(pd)
        mra = whitening(mra)

        # Create a 4D multi-sequence image (i.e. [channels, x, y, z])
        images = np.asarray([t1, t2, pd, mra]).astype(np.float32)

        # Transpose to [batch, x, y, z, channel] as required input by the network
        images = np.transpose(images, (1, 2, 3, 0))

        if mode == tf.estimator.ModeKeys.PREDICT:
            yield {'features': {'x': images}}

        # Augment if used in training mode
        if mode == tf.estimator.ModeKeys.TRAIN:
            images, _ = _augment(images)
        
        # Check if the reader is supposed to return training examples or full images
        if params['extract_examples']:
            images, _ = extract_random_example_array(images, example_size=params['example_size'], n_examples=params['n_examples'])
            
            for e in range(params['n_examples']):
                yield {'features': {'x': images[e].astype(np.float32)}}
        else:
            yield {'features': {'x': images}}

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