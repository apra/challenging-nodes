FROM nvidia/cuda:11.1.1-devel-ubuntu18.04

ENV CUDA_ROOT /usr/local/cuda/bin
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64

RUN ldconfig && \
    apt-get -qq update && \
    apt-get update && \
    apt-get install -y libxext6 libsm6 libxrender1 libgl1-mesa-glx wget rsync tmux nano sudo dcmtk && \
    rm -rf /var/lib/apt/lists/* && \
    useradd -d /home/research -ms /bin/bash -g root -G sudo -p research research && \
    echo "research ALL=(root) NOPASSWD:ALL"  > /etc/sudoers.d/research && \
    chmod 0440 /etc/sudoers.d/research && \
    mkdir /cache && \
    chown research:root /cache

USER research
WORKDIR /home/research

RUN wget -q https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b

ENV PATH "/home/research/miniconda3/bin:$PATH:$CUDA_ROOT"
ENV DOCKER_CONFIG "true"
ENV TERM "screen-256color"

# Setup python packages
RUN conda install python=3.6.12 && \
    conda install pytorch torchvision tensorboard matplotlib tqdm flask=1.1.2 requests=2.24.0 pip pillow=8.0.1 && \
    pip install dill==0.3.4 opencv-python==4.5.3.56 opencxr

ENV PATH "/code/tools:$PATH"
ENV PYTHONPATH /code
WORKDIR /code

# Provide an open entrypoint for the docker
ENTRYPOINT $0 $@
