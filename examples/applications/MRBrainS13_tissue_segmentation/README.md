## Image segmentation of multi-channel brain MR images
Exemplary training scripts for tissue segmentation from multi-sequence (T1w, T1 inversion recovery, T2 Flair) brain MR images, based on the [MRBrainS13](http://mrbrains13.isi.uu.nl/) challenge data [1]. 

[1] AM Mendrik et al, (2015). MRBrainS challenge: online evaluation framework for brain image segmentation in 3T MRI scans. Computational intelligence and neuroscience.

![Exemplary segmentations](example.png)

### Data
The data can be downloaded [here](http://mrbrains13.isi.uu.nl/download.php) and requires registration. It includes 5 datasets and corresponding segmentations. CSV files contain the paths to the folders for the training and validation splits, respectively:

mrbrains.csv:
```
id, subj_folder
1, MY_DATA_PATH/MRBrainS13DataNii/TrainingData/1/
2, MY_DATA_PATH/MRBrainS13DataNii/TrainingData/2/
3, MY_DATA_PATH/MRBrainS13DataNii/TrainingData/3/
4, MY_DATA_PATH/MRBrainS13DataNii/TrainingData/4/
5, MY_DATA_PATH/MRBrainS13DataNii/TrainingData/5/
```

These are parsed and extract tf.Tensor examples for training and evaluation in `reader.py` using a [SimpleITK](http://www.simpleitk.org/) for i/o of the .nii files:

```
...
t1 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(str(img_fn), 'T1.nii')))
t1_ir = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(str(img_fn), 'T1_IR.nii')))
...
```

### Training
![Dice_similarity_coefficient](dsc.png)

To train a new model, run the train.py script:

  ```
  python -u train.py MY_OPTIONS
  ```

### Monitoring

For monitoring and metric tracking, spawn a tensorboard webserver and point the log directory to the model_path:

  ```
  tensorboard --logdir /tmp/mrbrains_segmentation/
  ```
  
### Deploy

To deploy a model and run inference, run the deploy.py script and point to the model_path:

  ```
  python -u deploy.py --model_path /tmp/mrbrains_segmentation/ MY_OPTIONS
  ```