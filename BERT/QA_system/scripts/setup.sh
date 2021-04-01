#!/usr/bin/bash
set -e 

pip install pip-tools
pip-compile requirements.in
pip-sync

git clone https://github.com/huggingface/transformers
cd transformers
pip install --editable .


