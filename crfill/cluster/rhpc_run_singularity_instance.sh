#!/bin/bash

JOBS_SOURCE="$HOME/projects/crfill"
SINGULARITYIMAGE="$HOME/projects/singularity_files/crfill.sif"
SINGULARITY_NAME="CRFILL"
DATA="$SCRATCH"

LOGGING_DIR="$HOME/projects/crfill/checkpoints"


singularity instance start --no-home --nv \
--bind "$DATA":/data \
--bind $LOGGING_DIR:"$LOGGING_DIR" \
--bind "$JOBS_SOURCE" \
--bind "$DATA" \
$SINGULARITYIMAGE \
$SINGULARITY_NAME