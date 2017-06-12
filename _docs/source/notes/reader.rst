Reader
======

DLTK includes ``readers`` to provide data reading functionality. Those readers build queues to dynamically load data
from the disk into memory. To implement a custom reader for your data you can simply inherit from the
:class:`~dltk.core.io.reader.AbstractReader` class and overwrite the ``__init__`` and ``_read_sample`` functions. An
example for a custom reader class is the ``MRBrains`` reader which reads NIFTI files and returns class-balanced patches
for training a segmentation network.

This reader doesn't need to overwrite the ``__init__`` function as there are no additional parameters to store. It then
overwrites the ``_read_sample`` function which takes paths of directores with the MRBrains files and reads them with
`SimpleITK <https://itk.org/Wiki/SimpleITK/GettingStarted>`_.
::
  class MRBrainsReader(AbstractReader):
    """
    A custom reader for MRBrainS13 with online preprocessing and augmentation

    """
    def _read_sample(self, id_queue, n_examples=1, is_training=True):
        """
        A read function for nii images using SimpleITK

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

For convention the reader uses the ``_preprocess`` and ``_augment`` function to implement data preprocessing and
augmentation. For preprocessing the reader simply whitens each read NIFTI image independently
::
      def _preprocess(self, data):
        """ Simple whitening """
        return [whitening(img) for img in data]

and then augments the stacked image by extracting class balanced patches, random flipping of the patches along the
sagittal axis as well as gaussian noise and offset.
::
      def _augment(self, img, lbl, n_examples):
        """
            Data augmentation during training

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

The :class:`~dltk.core.io.reader.AbstractReader` class wraps the ``_read_sample`` function to fill a Tensorflow queue.
To use this custom reader you must build an instance of the reader and specify the returned data types and returned
shapes of the tensor.
::
  reader = MRBrainsReader([tf.float32, tf.int32], [[24, 64, 64, 3], [24, 64, 64]], name='train_queue')

You have then access to the queue by calling the reader object with a list of directories to read from
::
  x_train, y_train = reader(train_files)

If your custom reader is less flexible you can also set the data types and data shapes in the ``__init__`` function like
::
  def __init__(self):
    dtypes = your_dtypes
    dshapes = your_dshapes
    name = 'your_reader_name'
    super(cls, self).__init__(dtypes, dshapes, name)