#!/usr/bin/env bash
set -x

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


# Note on skipping axlearn/experiments/golden_config_test.py: abseil https://github.com/abseil/abseil-py/blob/7b4b0e29389e5b8ce4b4b5d39d7aea4c30967b87/absl/testing/parameterized.py#L343C29-L343C44 doesnt' expect
#     ParameterSet https://github.com/pytest-dev/pytest/blob/d0f136fe64f9374f18a04562305b178fb380d1ec/src/_pytest/mark/__init__.py#L72 so we can't mark it, but since test_run (the only test that runs workload) is ran manually it is not an issue.
#     We have to run manually this script and review all the skips.
#neuron and GPU single worker
# pytest --durations=0 --junit-xml=report.xml --timeout=900 -vvv -s -n 1 \
#   -m "not (gs_login or tpu or high_cpu or fp64) and not (inference or vision or moe or multimodal or audio or speech)" \
#   --ignore=axlearn/cloud --ignore=axlearn/open_api --ignore=axlearn/experiments/golden_config_test.py .

#CPU multi worker
pytest --durations=0 --junit-xml=report.xml --timeout=900 -vvv -s -n auto --dist worksteal \
  -m "not (gs_login or tpu or high_cpu or fp64) and not (inference or vision or moe or multimodal or audio or speech)" \
  --ignore=axlearn/cloud --ignore=axlearn/open_api \
  --ignore=axlearn/experiments/vision --ignore=axlearn/experiments/audio \
  --ignore=axlearn/audio --ignore=axlearn/vision \
  --ignore=axlearn/experiments/golden_config_test.py .