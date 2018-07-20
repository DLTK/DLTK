## Deep Learning Toolkit (DLTK) for Medical Imaging
[![Gitter](https://badges.gitter.im/DLTK/DLTK.svg)](https://gitter.im/DLTK/DLTK?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![Coverage Status](https://coveralls.io/repos/github/DLTK/DLTK/badge.svg?branch=master)](https://coveralls.io/github/DLTK/DLTK?branch=dev)
[![Build Status](https://travis-ci.org/DLTK/DLTK.svg?branch=master)](https://travis-ci.org/DLTK/DLTK)

![DLTK logo](logo.png)

DLTK is a neural networks toolkit written in python, on top of [TensorFlow](https://github.com/tensorflow/tensorflow). It is developed to enable fast prototyping with a low entry threshold and ensure reproducibility in image analysis applications, with a particular focus on medical imaging. Its goal is  to provide the community with state of the art methods and models and to accelerate research in this exciting field.

### Documentation
The DLTK API can be found [here](https://dltk.github.io/)

### Referencing and citing DLTK
If you use DLTK in your work please refer to this citation for the current version:

```
@article{pawlowski2017state,
  title={DLTK: State of the Art Reference Implementations for Deep Learning on Medical Images},
  author={Nick Pawlowski and S. Ira Ktena, and Matthew C.H. Lee and Bernhard Kainz and Daniel Rueckert and Ben Glocker and Martin Rajchl},
  journal={arXiv preprint arXiv:1711.06853},
  year={2017}
}
```

If you use any application from the [DLTK Model Zoo](https://github.com/DLTK/models), additionally refer to the respective README.md files in the applications' folder to comply with its authors' instructions on referencing.

### Introduction to Biomedical Image Analysis
To ease into the subject, we wrote a quick overview [blog entry](https://medium.com/tensorflow/an-introduction-to-biomedical-image-analysis-with-tensorflow-and-dltk-2c25304e7c13) (12 min read) for the new [TensorFlow blog](https://medium.com/tensorflow). It covers some of the speciality information required for working with medical images and we suggest to read it, if you are new to the topic. The code we refer to in the blog can be found in [examples/tutorials](https://github.com/DLTK/DLTK/tree/master/examples/tutorials) and [examples/applications](https://github.com/DLTK/DLTK/tree/master/examples/applications).

### Installation
1. Setup a virtual environment and activate it. Although DLTK<=0.2.1 supports and python 2.7, we will not support it future releases, similarly to our dependencies (i.e. SciPy, NumPy). We highly recommend using python3. If you intend to run this on machines with different system versions, use the --always-copy flag:

   ```shell
   virtualenv -p python3 --always-copy venv_tf
   source venv_tf/bin/activate
   ```
   
2. Install TensorFlow (>=1.4.0) (preferred: with GPU support) for your system
 as described [here](https://www.tensorflow.org/install/):
   
   ```shell
   pip install tensorflow-gpu>=1.4.0
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
   In ```examples/tutorials``` you will find tutorial notebooks to better understand on how DLTK interfaces with TensorFlow, how to write custom read functions and how to write your own ```model_fn```.   
   
   To run a notebook, navigate to the DLTK source root folder and open a notebook server on ```MY_PORT``` (default 8888):
   
   ```shell
   cd MY_WORKSPACE_DIRECTORY/DLTK
   jupyter notebook --ip=* --port MY_PORT
   ```

   Open a browser and enter the address ```http://localhost:MY_PORT``` or ```http://MY_DOMAIN_NAME:MY_PORT```. You can then navigate to a notebook in ```examples/tutorials```, open it (c.f. extension .ipynb) and modify or run it.

3. Example applications
   There are several example applications in ```examples/applications``` using the data in 1. Each folder contains an experimental setup with an application. **Please note that these are not tuned to high performance, but rather to showcase how to produce functioning scripts with DLTK models.** For additional notes and expected results, refer to the notes in the individual example's README.md.  

### DLTK Model Zoo
We also provide a zoo with (re-)implementations of current research methodology in a separate repository [DLTK/models](https://github.com/DLTK/models). Each model in the zoo is maintained by the respective authors and implementations often differ to those in ```examples/applications```. For instructions and information on the individual application in the zoo, please refer to the respective README.md files.

### How to contribute
We appreciate any contributions to the DLTK and its Model Zoo. If you have improvements, features or patches, please send us your pull requests! You can find specific instructions on how to issue a PR on github [here](https://help.github.com/articles/about-pull-requests/). Feel free to open an [issue](https://github.com/DLTK/DLTK/issues) if you find a bug or directly come chat with us on our gitter channel [![Gitter](https://badges.gitter.im/DLTK/DLTK.svg)](https://gitter.im/DLTK/DLTK?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge).

#### Basic contribution guidelines
- Python coding style: Like TensorFlow, we loosely adhere to [google coding style](https://google.github.io/styleguide/pyguide.html) and [google docstrings](https://google.github.io/styleguide/pyguide.html#Comments).
- Entirely new features should be committed to ```dltk/contrib``` before we can sensibly integrate it into the core.
- Standalone problem-specific applications or (re-)implementations of published methods should be committed to the [DLTK Model Zoo](https://github.com/DLTK/models) repo and provide a README.md file with author/coder contact information. 

#### Running tests locally
To run the tests on your machine, you can install the ``tests`` extras by 
running `pip install -e '.[tests]'` inside the DLTK root directory. This 
will install all necessary dependencies for testing. You can then run 
`pytest --cov dltk --flake8 --cov-append` to see whether your code passes.
 
#### Building docs locally
To run the tests on your machine, you can install the ``docs`` extras by 
running `pip install -e '.[docs]'` inside the DLTK root directory. This 
will install all necessary dependencies for the documentation. You can then run 
`make -C docs html` to build the documentation. You can access this 
documentation in a webbrowser of your choice by pointing it at 
`docs/build/html/index.html`.
 
### The team
DLTK is currently maintained by [@pawni](https://github.com/pawni) and [@mrajchl](https://github.com/mrajchl) with greatly appreciated contributions coming from individual researchers and engineers listed here in alphabetical order: 
[@CarloBiffi](https://github.com/CarloBiffi) [@ericspod](https://github.com/ericspod) [@ghisvail](https://github.com/ghisvail) [@mauinz](https://github.com/mauinz) [@michaeld123](https://github.com/michaeld123) [@sk1712](https://github.com/sk1712)

### License
See [LICENSE](https://github.com/DLTK/DLTK/blob/master/LICENSE)

### Acknowledgments
We would like to thank [NVIDIA GPU Computing](http://www.nvidia.com/) for providing us with hardware for our research. 
