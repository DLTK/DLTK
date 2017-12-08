# Docker support

A simple implementation of docker. The Dockerfile will build an docker image based on tensorflow 1.4.0-py3, or whichever is stated in the Dockerfile. By default it will start a jupyter notebook with tensorflow and DLTK tutorials available. To avoid downloading DLTK example data multiple times (70GB+), keep your data folder persistent, see below.

### Intructions
Prerequisites: docker

#### Build image
From tools/docker, run

`docker build -t tensorflow-dltk .`

#### Run notebook
`docker run --rm -it -p 8888:8888 tensorflow-dltk`

Note that no data or notebooks will be persistent

#### Run notebook with persistent data
`docker run --rm -it -v /path/to/DTLK/data:/data -p 8888:8888 tensorflow-dltk`

You data will be mounted at /data.

#### Run notebook with persistent data and notebooks
`docker run --rm -it -v /path/to/mydata:/data  -v /path/to/own/notebooks:/notebooks/your-notebook-folder-name  -p 8888:8888 tensorflow-dltk`

You data will be mounted at /data, and notebooks will be available in the Jupyter Notebook.

#### Run bash
`docker run --rm -it tensorflow-dltk bash`

#### CUDA/GPU SUPPORT
Prerequisites: nvidia-docker

In Dockerfile, change the FROM image to one of the gpu tagged images from https://hub.docker.com/r/tensorflow/tensorflow/tags/
Rebuild the image, and run by using `nvidia-docker` instaed of `docker` with any of the examples above.

#### Examples
Non-official example images can be found at https://hub.docker.com/r/mdjaere/tensorflow-dltk/
