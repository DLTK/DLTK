## DLTK Examples


### Data
Please refer to $DLTK_SRC/data for instructions on downloading publicly available datasets. These scripts also include some degree of preprocessing (creating csvs, resampling image data, etc). 


### Tutorials
The tutorials will contain jupyter notebooks to quickly learn the mechanics of DLTK and to start playing right away:

1. Start a notebook server with
```shell
cd $DLTK_SRC
jupyter notebook --ip=* --port $MY_PORT
```
 
2a. Navigate to tutorials and browse through the notebooks 

2b. Download the required dataset (see Data) and run the notebook from scratch


### Applications
More complete applications (i.e. representation learning, image segmentation, etc) can be found in the examples folder. Each folder contains an experimental setup with an application. Please note that these are not tuned to high performance, but rather to showcase how to produce functioning scripts with DLTK models.