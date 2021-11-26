FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04

# Based on https://github.com/anurag/fastai-course-1/

# Find the release at https://repo.anaconda.com/miniconda/
# Python 3.7
#ARG PYTHON_VERSION=3.7
#ARG CONDA_VERSION=Miniconda3-py37_4.9.2-Linux-x86_64.sh

# Python 3.9
ARG PYTHON_VERSION=3.9
ARG CONDA_VERSION=Miniconda3-py39_4.9.2-Linux-x86_64.sh
ARG CONDA_DIR=/opt/conda
ARG USERNAME=docker
ARG USERID=1000

ENV TZ=Europe
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Instal basic utilities
RUN apt-get update && \
  apt-get install -y --no-install-recommends git wget unzip bzip2 sudo build-essential ffmpeg libsm6 libxext6 && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV PATH $CONDA_DIR/bin:$PATH
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && \
  wget https://repo.continuum.io/miniconda/$CONDA_VERSION -O /tmp/miniconda.sh && \
  echo 'export PATH=$CONDA_DIR/bin:$PATH' > /etc/profile.d/conda.sh && \
  /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
  rm -rf /tmp/* && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# Create the user
RUN useradd --create-home -s /bin/bash --no-user-group -u $USERID $USERNAME && \
  chown $USERNAME $CONDA_DIR -R && \
  adduser $USERNAME sudo && \
  echo "$USERNAME ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

USER $USERNAME
WORKDIR /home/$USERNAME

RUN conda install -y python=$PYTHON_VERSION && \
  conda install -y pytorch torchvision cudatoolkit=11.3 -c pytorch && \
  conda install -y h5py scikit-learn matplotlib seaborn \
  pandas mkl-service cython tqdm flask=1.1.2 requests=2.24.0 pillow=8.0.1 && \
  conda clean -tipy

RUN  pip install --upgrade pip && \
  pip install pillow-simd dill==0.3.4 opencv-python==4.5.4.60 && \
  rm -rf ~/.cache/pip

ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_ROOT=$CUDA_HOME
ENV PATH=$PATH:$CUDA_ROOT/bin:$HOME/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_ROOT/lib64