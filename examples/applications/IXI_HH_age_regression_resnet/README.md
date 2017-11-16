## Age regression from 3T T1w brain MR images
Exemplary training and evaluation scripts for regression from T1w brain MR images, based on the [IXI dataset](http://brain-development.org/ixi-dataset/) [1]. 

[1] IXI – Information eXtraction from Images (EPSRC GR/S21533/02)

### Data
The data can be downloaded via the script in dltk/data/IXI_HH. It includes 178 datasets and corresponding demographic information. The download script
 - produces a CSV file containing demographic information
 - validates the completeness of all imaging data for each database entry
 - resamples the images to 1mm isotropic resolution
 - removes .tar files and original images

demographic_HH.csv:
```
IXI_ID,"SEX_ID (1=m, 2=f)",HEIGHT,WEIGHT,ETHNIC_ID,MARITAL_ID,OCCUPATION_ID,QUALIFICATION_ID,DOB,DATE_AVAILABLE,STUDY_DATE,AGE
IXI012,1,175,70,1,2,1,5,1966-08-20,1,2005-06-01,38.7816563997
IXI013,1,182,70,1,2,1,5,1958-09-15,1,2005-06-01,46.7104722793
...
```

In `train.py`, the CSV is parsed and split into a training and validation set. A custom `reader.py` extracts tf.Tensor examples for training and evaluation in using a [SimpleITK](http://www.simpleitk.org/) for  i/o of the .nii files:

```
...
t1 = sitk.GetArrayFromImage(sitk.ReadImage(t1_fn))
t2 = sitk.GetArrayFromImage(sitk.ReadImage(t2_fn))
pd = sitk.GetArrayFromImage(sitk.ReadImage(pd_fn))
...
```

### Notes 
In this example we use the first 150 datasets for training, the rest for validation. Here are some quick statistics on the sets:

All subjects:
Age: mean = 47.35, sd = 16.76, min = 20.17, max = 81.94

Training subjects:
Age: mean = 48.00, sd = 17.14, min = 20.17, max = 81.94

Evaluation subjects:
Age: mean = 43.89, sd = 14.02, min = 25.53, max = 71.21


### Usage
- To train a new model, run the train.py script:

  ```
  python -u train.py MY_OPTIONS
  ```

  The model and training events will be saved to a temporary folder: `/tmp/IXI_regression`.

- For monitoring and metric tracking, spawn a tensorboard webserver and point the log directory to the model save_path:

  ```
  tensorboard --logdir /tmp/IXI_regression/
  ```

- To deploy a model and run inference, run the deploy.py script and point to the model save_path:

  ```
  python -u deploy.py --model_path /tmp/IXI_regression MY_OPTIONS
  ```
  
  This should result in an output similar to this:  
  ```
  id=IXI566; pred=47.20 yrs; true=42.97 yrs; run time=1.75 s;   
  id=IXI567; pred=33.68 yrs; true=28.56 yrs; run time=0.32 s; 
  ...
  mean absolute err=5.399
  ```
   


