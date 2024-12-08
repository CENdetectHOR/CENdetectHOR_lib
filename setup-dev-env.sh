#!/usr/bin/env bash

ENV_DIR="./env"

conda init --all
source ~/.zshrc

if [ ! -d "$ENV_DIR" ]; then
    conda create -p "$ENV_DIR" -y python=3.9
fi

conda activate $ENV_DIR
python -m pip install -e .
conda deactivate