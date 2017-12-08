# Base image from https://hub.docker.com/r/tensorflow/tensorflow/tags/
FROM tensorflow/tensorflow:1.4.0-py3
# Move tensorflow tutorials into its own folder
RUN mv /notebooks /tensorflow-tutorials && mkdir /notebooks && mv /tensorflow-tutorials /notebooks
# Get DLTK master from github
RUN curl -LOk https://github.com/DLTK/DLTK/archive/master.tar.gz && tar xzf master.tar.gz && mv DLTK-master /opt/DLTK && rm master.tar.gz
WORKDIR /opt/DLTK
# Install DLTK from source
RUN pip install --no-cache-dir -e .
# Copy DLTK tutorials into notebook folder
RUN cp -r ./examples/tutorials /notebooks/DLTK-tutorials && cp -r ./data /data
# TensorBoard
EXPOSE 6006
# IPython
EXPOSE 8888
WORKDIR "/notebooks"
CMD ["/run_jupyter.sh", "--allow-root"]
