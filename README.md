## Deep Learning Toolkit (DLTK)
[![Gitter](https://badges.gitter.im/DLTK/DLTK.svg)](https://gitter.im/DLTK/DLTK?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

DLTK is a neural networks toolkit written in python, on top of [Tensorflow](https://github.com/tensorflow/tensorflow). It is developed to enable fast prototyping with a low entry threshold and ensure reproducibility in image analysis applications, with a particular focus on medical imaging. Its goal is  to provide the community with state of the art methods and models and to accelerate research in this exciting field.

### Documentation
DLTK API and user guides can be found [here](https://dltk.github.io/)


### TODO v0.2:
- deploy
- fine tuning
- histogram estimation mit moving averages session hook domain adaption
- hooks for sampling
- resampling etc
- validation/inference on full volumes
- json config?
- CI / pytest
- in notebook training
- setting the example shape globally (i.e. for summaries and for the reader)

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

3a. Clone the source and install DLTK (including tensorflow) via pip in edit mode:

```shell
git clone https://github.com/DLTK/DLTK.git $DLTK_SRC
cd $DLTK_SRC
pip install -e .
```

3b. Install DLTK directly from pypi as is:
```shell
pip install dltk
```


### Start playing
1. Start a notebook server with
```shell
cd $DLTK_SRC
jupyter notebook --ip=* --port $MY_PORT
```
 
2. navigate to examples and run a tutorial.ipynb notebook 

    
### Dev team
[@mrajchl](https://github.com/mrajchl)
[@pawni](https://github.com/pawni)

### Contributors
[@ericspod](https://github.com/ericspod)
[@ghisvail](https://github.com/ghisvail)
[@mauinz](https://github.com/mauinz)
[@michaeld123](https://github.com/michaeld123)
[@sk1712](https://github.com/sk1712)

### License
See license.md

### Acknowledgements
We would like to thank [NVIDIA GPU Computing](http://www.nvidia.com/) for providing us with hardware for our research. 



