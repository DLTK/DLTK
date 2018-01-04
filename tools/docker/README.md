# DLTK Docker

Docker image with DLTK and tensorflow, and Jupyter Notebook tutorials. To avoid re-downloading DLTK example data(70GB+), make sure to keep the data folder persistent, see below. For more info, see [Docker documenation](https://docs.docker.com/).

### Intructions
Prerequisites: [docker](https://www.docker.com/)

#### Pull image
`docker pull dltk/dltk`

#### Run Notebook
`docker run --rm -it -p 8888:8888 dltk/dltk`

No data or notebooks will be persistent.

#### Run Notebook with persistent data
`docker run --rm -it -v /path/to/DTLK/data:/data -p 8888:8888 dltk/dltk`

You persistent data is available at `/data`. Notebooks will not be persistent.

#### Run Notebook with persistent data and notebooks
`docker run --rm -it -v /path/to/mydata:/data  -v /path/to/own/notebooks:/notebooks/your-notebook-folder-name  -p 8888:8888 dltk/dltk`

You persistent data is available at `/data`. Your own notebooks will be persistent. Tutorials will not be persisent.

#### Note: Setting Notebook password
By default Jupyter Notebook will create access token for login. To set a password, set environment variable `PASSWORD` at container deployment, for example run docker with `-e PASSWORD=password1234`.

#### Run bash
`docker run --rm -it dltk/dltk bash`

#### Build image from Dockerfile
From folder `tools/docker`, run `docker build -t dltk/dltk:yourbuildtag .` Run with `docker run --rm -it dltk/dltk:yourbuildtag`.

Tip: Change tensorflow version by editing the Dockerfile base image tag to one from https://hub.docker.com/r/tensorflow/tensorflow/tags/ (includes release candidates, nightlies, gpu, etc).

#### CUDA/GPU SUPPORT
Prerequisites: [nvidia-docker](https://github.com/NVIDIA/nvidia-docker/)

Change the Dockerfile base image to one of the gpu tagged images from https://hub.docker.com/r/tensorflow/tensorflow/tags/. Build the image, and run docker with `--runtime=nvidia`. See https://github.com/NVIDIA/nvidia-docker for more info.
