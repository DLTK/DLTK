from __future__ import absolute_import
from __future__ import print_function

import os

import SimpleITK as sitk
import numpy as np
import tensorflow as tf

from dltk.core.io.preprocessing import *
from dltk.core.io.augmentation import *
from dltk.core.io.reader import AbstractReader


class MRBrainsReader(AbstractReader):
    """
    A custom reader for MRBrainS13 with online preprocessing and augmentation
        
    """
    def _read_sample(self, id_queue, n_examples=1, is_training=True):
        """A read function for nii images using SimpleITK

        Parameters
        ----------
        id_queue : str
            a file name or path to be read by this custom function
        
        n_examples : int
            the number of examples to produce from the read image
        
        is_training : bool
        
        Returns
        -------
        list of np.arrays of image and label pairs as 5D tensors
        """

        path_list = id_queue[0]

        # Use a SimpleITK reader to load the multi channel nii images and labels for training 
        t1 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(str(path_list[0]), 'T1.nii')))
        t1_ir = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(str(path_list[0]), 'T1_IR.nii')))
        t2_fl = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(str(path_list[0]), 'T2_FLAIR.nii')))
        lbl = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(str(path_list[0]), 'LabelsForTraining.nii')))

        # Create a 5D image array with dimensions [batch_size, x, y, z, channels] and preprocess them
        data = self._preprocess([t1, t1_ir, t2_fl])
        img = np.transpose(np.array(data), (1, 2, 3, 0))
        if is_training:
            img, lbl = self._augment(img, lbl, n_examples)

        return [img, lbl]

    def _preprocess(self, data):
        """ Simple whitening """
        return [whitening(img) for img in data]

    def _augment(self, img, lbl, n_examples):
        """Data augmentation during training

        Parameters
        ----------
        img : np.array
            a 4D input image with dimensions [x, y, z, channels]

        lbl : np.array
            a 3D label map corresponding to img

        n_examples : int
            the number of examples to produce from the read image

        Returns
        -------
        list of np.arrays of image and label pairs as 5D tensors
        """
        
        # Extract training example image and label pairs
        imgs, lbls = extract_class_balanced_example_array(img, lbl, self.dshapes[0][:-1], n_examples, 9)
        
        # Randomly flip the example pairs along the sagittal axis
        for i in range(imgs.shape[0]):
            tmp_img, tmp_lbl = flip([imgs[i], lbls[i]], axis=2)
            imgs[i] = tmp_img
            lbls[i] = tmp_lbl
            
        # Add Gaussian noise and offset to the images
        imgs = gaussian_noise(imgs, sigma=0.2)
        imgs = gaussian_offset(imgs, sigma=0.5)
        return imgs, lbls