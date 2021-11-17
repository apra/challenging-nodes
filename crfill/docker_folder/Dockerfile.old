FROM nvidia/cuda:11.1.1-devel-ubuntu18.04

ENV CUDA_ROOT /usr/local/cuda/bin
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64

RUN ldconfig
RUN apt-get -qq update
# libsm6 and libxext6 are needed for cv2
RUN apt-get update && apt-get install -y libxext6 libsm6 libxrender1 libgl1-mesa-glx git wget rsync tmux nano vim sudo dcmtk && \
  rm -rf /var/lib/apt/lists/*

# Make a user
RUN useradd -d /home/unet -ms /bin/bash -g root -G sudo -p unet unet
RUN echo "unet ALL=(root) NOPASSWD:ALL"  > /etc/sudoers.d/unet && \
  chmod 0440 /etc/sudoers.d/unet

RUN mkdir /cache
RUN chown unet:root /cache

USER unet
WORKDIR /home/unet

RUN wget -q https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b

ENV PATH "/home/unet/miniconda3/bin:$PATH:$CUDA_ROOT"
ENV DOCKER_CONFIG "true"
ENV TERM "screen-256color"

# Setup python packages
RUN conda install python=3.6.12
RUN conda install pytorch torchvision tensorboard matplotlib tqdm flask=1.1.2 requests=2.24.0 pip pillow=8.0.1
RUN pip install dill==0.3.4 opencv-python==4.5.3.56 opencxr

ENV PATH "/code/tools:$PATH"
ENV PYTHONPATH /code
WORKDIR /code

# Provide an open entrypoint for the docker
ENTRYPOINT $0 $@
