FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN python -m pip install --user -U pip
RUN  pip install --upgrade pip

# Copy all required files so that they are available within the docker image
# All the codes, weights, anything you need to run the algorithm!
COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
#COPY --chown=algorithm:algorithm entrypoint.sh /opt/algorithm/
COPY --chown=algorithm:algorithm model_submission /opt/algorithm/model_submission

# Install required python packages via pip - please see the requirements.txt and adapt it to your needs
RUN python -m pip install --user -rrequirements.txt

COPY --chown=algorithm:algorithm process.py /opt/algorithm/

# Entrypoint to run, entypoint.sh files executes process.py as a script
ENTRYPOINT python -m process $0 $@
#ENTRYPOINT ["bash", "entrypoint.sh"]

## ALGORITHM LABELS ##

# These labels are required
LABEL nl.diagnijmegen.rse.algorithm.name=nodulegeneration

# These labels are required and describe what kind of hardware your algorithm requires to run.
LABEL nl.diagnijmegen.rse.algorithm.hardware.cpu.count=4
LABEL nl.diagnijmegen.rse.algorithm.hardware.cpu.capabilities=()
LABEL nl.diagnijmegen.rse.algorithm.hardware.memory=40G
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.count=1
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.cuda_compute_capability=
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.memory=11G
