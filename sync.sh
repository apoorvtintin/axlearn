rsync -av \
  axlearn/experiments/text/gpt/fuji.py \
  axlearn/experiments/text/gpt/c4_trainer.py \
  run.sh \
  axlearn/common/flash_attention/utils.py \
  axlearn/common/flash_attention/neuron_attention.py \
  axlearn/common/attention_bias.py \
  axlearn/common/input_lm.py \
  axlearn/common/input_tf_data.py \
  fshead:/fsx/thangakr/override_dnm2/

rsync -av \
  run.slurm \
  fshead:/fsx/thangakr/run.dnm2.slurm
