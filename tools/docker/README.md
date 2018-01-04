# DLTK Docker

Docker image with DLTK and tensorflow, and Jupyter Notebook tutorials. To avoid downloading the DLTK example data multiple times (70GB+), make sure to keep it persistent, see below. For more info, see [Docker documenation](https://docs.docker.com/).

### Intructions
Prerequisites: [docker](https://www.docker.com/)

#### Pull image
`docker pull dltk/dltk`

#### Run Notebook
`docker run --rm -it -p 8888:8888 dltk/dltk`

No data or notebooks will be persistent.

#### Run Notebook with persistent DLTK data
`docker run --rm -it -v /path/to/DTLK/data:/data -p 8888:8888 dltk/dltk`

#### Run Notebook with persistent data and notebooks
`docker run --rm -it -v /path/to/own/notebook/folder:/notebooks/your-notebook-folder -p 8888:8888 dltk/dltk`

Notebooks and data in your notebook folder will be persistent. Tutorials will not be persistent.

#### Note: Setting Notebook password
By default Jupyter Notebook will create access token for login. To set a password, set environment variable `PASSWORD` at container deployment, for example `docker run -e PASSWORD=password1234 …`.

#### Run bash
`docker run --rm -it dltk/dltk bash`

#### Build image from Dockerfile
From folder `tools/docker`, run `docker build -t dltk/dltk:yourbuildtag .` Run with `docker run --rm -it dltk/dltk:yourbuildtag`.

Tip: You can change tensorflow version by editing the Dockerfile base image tag to one from https://hub.docker.com/r/tensorflow/tensorflow/tags/, which includes release candidates, nightlies, gpu, etc. NB. DLTK is not guaranteed to run with all versions.

#### CUDA/GPU SUPPORT
Prerequisites: [nvidia-docker](https://github.com/NVIDIA/nvidia-docker/)

To add GPU support, edit the Dockerfile base image to one of the gpu tagged images from https://hub.docker.com/r/tensorflow/tensorflow/tags/. Rebuild the image, and run with `docker run --runtime=nvidia …`. See https://github.com/NVIDIA/nvidia-docker for more info.
