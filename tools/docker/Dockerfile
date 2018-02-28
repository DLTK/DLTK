# Base image from https://hub.docker.com/r/tensorflow/tensorflow/tags/
FROM tensorflow/tensorflow:1.4.0-py3

LABEL maintainer="Martin D. Jaere <martin.jaere11@imperial.ac.uk>"

# Move tensorflow tutorials into folder
RUN mv /notebooks /tensorflow-tutorials && \
    mkdir /notebooks && \
    mv /tensorflow-tutorials /notebooks

# Pull master branch from github
RUN curl -LOk https://github.com/DLTK/DLTK/archive/master.tar.gz && \
    tar xzf master.tar.gz && \
    mv DLTK-master /opt/DLTK && \
    rm master.tar.gz

# Install from source
RUN pip install --no-cache-dir -e /opt/DLTK

# Copy DLTK tutorials into notebook folder
RUN cp -r /opt/DLTK/examples/tutorials /notebooks/DLTK-tutorials

# TensorBoard
EXPOSE 6006

# IPython
EXPOSE 8888
WORKDIR /notebooks

# Copy dl instruction files into mounted /data to allow for data persistense
# Run notebook
CMD cp -r -n /opt/DLTK/data/* /data && /run_jupyter.sh --allow-root
