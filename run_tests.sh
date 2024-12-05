#!/usr/bin/env bash

set -e -x

# Install the package (necessary for CLI tests).
# Requirements should already be cached in the docker image.
pip install -e .

# Log installed versions
pip freeze

#apt list


download_assets() {
  set -e -x
  mkdir -p axlearn/data/tokenizers/sentencepiece
  mkdir -p axlearn/data/tokenizers/bpe
  curl https://huggingface.co/t5-base/resolve/main/spiece.model -o axlearn/data/tokenizers/sentencepiece/t5-base
  curl https://huggingface.co/FacebookAI/roberta-base/raw/main/merges.txt -o axlearn/data/tokenizers/bpe/roberta-base-merges.txt
  curl https://huggingface.co/FacebookAI/roberta-base/raw/main/vocab.json -o axlearn/data/tokenizers/bpe/roberta-base-vocab.json
}

download_assets

pytest --durations=100 -v -n auto \
  -m "not (gs_login or tpu or high_cpu or fp64)" \
  --dist worksteal \
  --ignore=axlearn/cloud --ignore=axlearn/open_api  .
