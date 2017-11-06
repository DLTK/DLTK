## Deep Learning Toolkit (DLTK) for Medical Imaging
[![Gitter](https://badges.gitter.im/DLTK/DLTK.svg)](https://gitter.im/DLTK/DLTK?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

![DLTK logo](logo.png)

DLTK is a neural networks toolkit written in python, on top of [Tensorflow](https://github.com/tensorflow/tensorflow). It is developed to enable fast prototyping with a low entry threshold and ensure reproducibility in image analysis applications, with a particular focus on medical imaging. Its goal is  to provide the community with state of the art methods and models and to accelerate research in this exciting field.


### Documentation
The DLTK API can be found [here](https://dltk.github.io/)

### Installation
1. Install CUDA with cuDNN and add its path in ~/.bashrc by sourcing setup.sh:

   ```shell
   source MY_CUDA_PATH/setup.sh
   ```

2. Setup a virtual environment and activate it:

   ```shell
   virtualenv venv_tf
   source venv_tf/bin/activate
   ```

3. Install DLTK:
   There are two installation options available: You can simply install dltk as is from pypi via
   
   ```shell
   pip install dltk
   ```
   or you can clone the source and install DLTK in edit mode (preferred):

   ```shell
   cd MY_WORKSPACE_DIRECTORY
   git clone https://github.com/DLTK/DLTK.git 
   cd DLTK
   pip install -e .
   ```
   This will allow you to modify the actual DLTK source code and import that modified source wherever you need it via ```import dltk```. 


### Start playing

1. Downloading example data
   You will find download and preprocessing scripts for publicly available datasets in ```data```. To download the IXI HH dataset, navigate to ```data/IXI_HH``` and run the download script with ```python download_IXI_HH.py```.


2. Tutorial notebooks
   In ```examples/tutorials``` you will find tutorial notebooks to better understand on how DLTK interfaces with tensorflow, how to write custom read functions and how to write your own ```model_fn```.   
   
   To run a notebook, navigate to the DLTK source root folder and open a notebook server on ```MY_PORT``` (default 8888):
   
   ```shell
   cd MY_WORKSPACE_DIRECTORY/DLTK
   jupyter notebook --ip=* --port MY_PORT
   ```   
   Open a browser and enter the address ```http://localhost:MY_PORT``` or ```http://MY_DOMAIN_NAME:MY_PORT```. You can then navigate to a notebook in ```examples/tutorials```, open it (c.f. extension .ipynb) and modify or run it.

3. Example applications
   There are several example applications in ```examples/applications``` using the data in 1. Each folder contains an experimental setup with an application. Please note that these are not tuned to high performance, but rather to showcase how to produce functioning scripts with DLTK models. For additional notes and expected results, refer to the notes in the individual example's README.md. 

    
### Dev team
[@mrajchl](https://github.com/mrajchl)
[@pawni](https://github.com/pawni)

### Contributors (many thanks)
#### v0.1 [@ericspod](https://github.com/ericspod) [@ghisvail](https://github.com/ghisvail) [@mauinz](https://github.com/mauinz) [@michaeld123](https://github.com/michaeld123) [@sk1712](https://github.com/sk1712)

### License
See license.md

### Acknowledgements
We would like to thank [NVIDIA GPU Computing](http://www.nvidia.com/) for providing us with hardware for our research. 



