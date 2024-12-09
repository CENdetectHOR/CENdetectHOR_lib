#!/usr/bin/env bash

ENV_DIR="./test-env"

conda init --all
source ~/.zshrc

if [ ! -d "$ENV_DIR" ]; then
    conda create -p "$ENV_DIR" -y python=3.9
fi

conda activate $ENV_DIR
pip install cen_detect_hor
conda deactivate