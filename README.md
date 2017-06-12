## Deep Learning Toolkit (DLTK)
DLTK is a neural networks toolkit written in python, on top of [Tensorflow](https://github.com/tensorflow/tensorflow). Its modular architecture is closely inspired by [sonnet](https://github.com/deepmind/sonnet) and it was developed to enable fast prototyping and ensure reproducibility in image analysis applications, with a particular focus on medical imaging. Its goal is  to provide the community with state of the art methods and models and to accelerate research in this exciting field.


### Installation
1. Install CUDA with cuDNN and add the path to ~/.bashrc:

```shell
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:MY_CUDA_PATH/lib64; export LD_LIBRARY_PATH
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:MY_CUDA_PATHextras/CUPTI/lib64; export LD_LIBRARY_PATH
PATH=$PATH:MY_CUDA_PATH/bin; export PATH
CUDA_HOME=MY_CUDA_PATH; export CUDA_HOME
```

2. Setup a virtual environment and activate it:

```shell
virtualenv venv_tf1.1
source venv_tf1.1/bin/activate
```

3. Install all DLTK dependencies (including tensorflow) via pip:

```shell
cd $DLTK_SRC
pip install -e .
```

### Start playing
1. Start a notebook server with
```shell
jupyter notebook --ip=* --port $MY_PORT --no-browser
```
 
2. navigate to examples and run a tutorial.ipynb notebook 

### Road map 
Over the course of the next months we will add more content to DLTK. This road map outlines the immediate plans for what you will be seeing in DLTK soon:

* Core:
  * Losses: Dice loss, frequency reweighted losses, adversial training
  * Normalisation: layer norm, weight norm

* Models:
  * deepmedic
  * densenet
  * VGG
  * Super-resolution nets

* Other:
  * Augmentation via elastic deformations
  * Sampling with fixed class frequencies
  * Stand-alone deploy scripts
    
### Team
@m_rajchl 
@pawni
@sk1712
@mauinz

### License
See license.md

### Acknowledgements
We would like to thank [NVIDIA GPU Computing](http://www.nvidia.com/) for providing us with hardware for our research. 



