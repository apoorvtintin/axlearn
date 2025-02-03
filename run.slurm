#!/bin/bash
#SBATCH --job-name=thangakr
#SBATCH --output=/fsx/thangakr/slurm_logs/log_%j.out
#SBATCH --time=2:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive

# This script supports the following environment variables for configuration:
#
# JOB_DIR_PREFIX      - Base directory for job outputs. Default: /scratch/$U/jobs
#                       Special value "FSX" maps to /fsx/$U/jobs
# JAX_COMPILATION_CACHE_DIR - Directory for JAX compilation cache. 
#                            Default: $JOB_DIR/neuronxcc_cache
# N_LAYERS           - Number of model layers. Default: 4
# MAX_TRAIN_STEPS    - Maximum training steps. Default: 1000000
# BLOCK_ATTENTION    - Enable block attention. Default: 0
# PROFILE_AFTER_RUN  - Enable profiling. Default: 0
# INTERNAL_DEBUG_MODE - Debug mode setting. Default: "" (empty string)
# NEURON_RT_LOG_LEVEL_NRT - Neuron runtime log level. Default: ERROR
# LOAD_REFERENCE_CKPT - Load reference checkpoint. Default: unset
# SAVE_REFERENCE_CKPT - Save reference checkpoint. Default: unset

set -e

U="thangakr"

BASE=/home/ubuntu/$U
srun mkdir -p $BASE

# Set up base directories
SCRATCH_BASE=/scratch/$U
FSX_BASE=/fsx/$U
srun mkdir -p $SCRATCH_BASE

# Allow overriding job directory prefix via JOB_DIR_PREFIX env var
# Default to SCRATCH_BASE/jobs for faster local writes
# Use "FSX" as shortcut for /fsx/$U/jobs
if [ "$JOB_DIR_PREFIX" = "FSX" ]; then
    export JOB_DIR_PREFIX="/fsx/$U/jobs"
fi
export JOB_DIR_PREFIX=${JOB_DIR_PREFIX:-$SCRATCH_BASE/jobs}
export JOB_DIR="${JOB_DIR_PREFIX}/$SLURM_JOB_ID"
srun mkdir -p "$JOB_DIR"
srun mkdir -p $FSX_BASE/slurm_logs

# Set up compilation cache directory
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-"$JOB_DIR/neuronxcc_cache"}
srun mkdir -p "$JAX_COMPILATION_CACHE_DIR"

# Download run file
export RUNFILE="axlearn.dnm2.run"
srun /fsx/thangakr/download_runfile.sh

# Check driver version. Install if needed.
srun bash -c 'if ! apt list 2>/dev/null | grep -q "^aws-neuronx-dkms/now 2.x.4125.0 amd64 \[installed,local\]"; then sudo dpkg -i --force-all /fsx/thangakr/binaries/aws-neuronx-dkms_2.x.4125.0_amd64.deb; fi'
CHECK_STATUS=$?
if [ $CHECK_STATUS -ne 0 ]; then
    echo "Driver version check failed! Terminating job."
    exit 1
fi
srun sudo dpkg -i /fsx/thangakr/binaries/aws-neuronx-tools-2.0.10276.0.deb

# Set number of layers (default: 4)
export N_LAYERS=${N_LAYERS:-4}
export LNC=2
export USE_FSDP=1
export ARTIFACTS_PATH=${JOB_DIR}

# Load reference checkpoint if LOAD_REFERENCE_CKPT is set
if [ "$LOAD_REFERENCE_CKPT" = "1" ]; then
    echo "Loading checkpoint from S3"
    srun /fsx/thangakr/load_reference_checkpoint.sh
fi

# Source peak memory utility functions
source /fsx/thangakr/peak_mem_util.sh

write_zero_to_peaks

export NEURON_DUMP_PATH="/mnt/artifacts/neuron_dump"
export HLO_DUMP_PATH="/mnt/artifacts/hlo_dump"

export XLA_FLAGS="--xla_dump_hlo_as_text --xla_disable_hlo_passes=aws_neuron_flip_all_gather_dot,neuron-hierarchical-collectives --xla_dump_to=${HLO_DUMP_PATH} --xla_dump_hlo_pass_re='.*'"

# Neuron compiler flags
export NEURON_CC_FLAGS="--framework=XLA"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --internal-max-instruction-limit=20000000"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --target=trn2" # --distribution-strategy=llm-training"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --internal-num-neuroncores-per-sengine=2"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --model-type transformer"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --no-internal-hlo-remat"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --enable-mixed-precision-accumulation"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} -O1"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --tensorizer-options='--enable-hoist-fsdp-collectives'"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --internal-hlo2tensorizer-options='--remat-rope'"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --dump=${NEURON_DUMP_PATH}"


# Set defaults for profiling flags
export PROFILE_AFTER_RUN=${PROFILE_AFTER_RUN:-0}
export INTERNAL_DEBUG_MODE=${INTERNAL_DEBUG_MODE:-""}

# Enable profiling if requested
if [ "$PROFILE_AFTER_RUN" = "1" ]; then
    export INTERNAL_DEBUG_MODE="penguin"
    export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --internal-compiler-debug-mode=penguin"
fi

# Set maximum training steps (default: 1000000)
export MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-1000000}

# Set Neuron runtime log level (default: ERROR)
export NEURON_RT_LOG_LEVEL_NRT=${NEURON_RT_LOG_LEVEL_NRT:-ERROR}

# Set block attention flag (default: 0)
export BLOCK_ATTENTION=${BLOCK_ATTENTION:-0}

# Set sequence packing debug flag (default: 0)
export SEQ_PACK_DEBUG=${SEQ_PACK_DEBUG:-0}

# Set model config (default: fuji-70B-v2-flash)
export MODEL_CONFIG=${MODEL_CONFIG:-fuji-70B-v2-flash}

srun --label \
    ${SCRATCH_BASE}/${RUNFILE} \
    --rw --root \
    -m /fsx/thangakr/override_dnm2/trainer.py:/axlearn/axlearn/common/trainer.py \
    -m /fsx/thangakr/override_dnm2/attention_bias.py:/axlearn/axlearn/common/attention_bias.py \
    -m /fsx/thangakr/override_dnm2/input_tf_data.py:/axlearn/axlearn/common/input_tf_data.py \
    -m /fsx/thangakr/override_dnm2/fuji.py:/axltest/axltest/kaena.py \
    -m /fsx/thangakr/override_dnm2/c4_trainer.py:/axltest/axltest/c4_trainer.py \
    -m /fsx/thangakr/override_dnm2/run.sh:/axltest/axltest/run.sh \
    -m /fsx/thangakr/override_dnm2/utils.py:/axlearn/axlearn/common/flash_attention/utils.py \
    -m /fsx/thangakr/override_dnm2/neuron_attention.py:/axlearn/axlearn/common/flash_attention/neuron_attention.py \
    -m /fsx/thangakr/override_dnm2/input_lm.py:/axlearn/axlearn/common/input_lm.py \
    -m /fsx:/fsx \
    -e NEURON_RT_EXEC_TIMEOUT=180 \
    -e NEURON_RT_ENABLE_INTERNODE_EXECUTION_BARRIER=1 \
    -e NEURON_RT_DBG_DISABLE_POD=1 \
    -e NEURON_RT_IO_RING_CACHE_SIZE=0 \
    -e NEURON_RT_DBG_CC_DMA_PACKET_SIZE=4096 \
    -e NEURON_RT_DBG_DMA_PACKETIZATION_SIZE=104857 \
    -e NEURON_FSDP_NUM_LAYER_EARLY_AG_SHIFT=1 \
    -e NEURON_FSDP_NUM_LAYER_LATE_RS_SHIFT=2 \
    -e NEURON_CC_FLAGS \
    -e XLA_FLAGS \
    -e NEURON_RT_DBG_CC_INTER_MESH_MAX_NODES=0 \
    -e SAVE_EVERY_N_STEPS=3000 \
    -e KEEP_LAST_N=3 \
    -e NEURON_ENABLE_INT_MATMUL_DOWNCAST=1 \
    -e NEURON_RT_DBG_MESH_CC=0 \
    -e BLOCK_ATTENTION \
    -e PROFILE_AFTER_RUN \
    -e INTERNAL_DEBUG_MODE \
    -e JAX_COMPILATION_CACHE_DIR \
    -e MAX_TRAIN_STEPS \
    -e NEURON_RT_LOG_LEVEL_NRT \
    -e N_LAYERS \
    -e SEQ_PACK_DEBUG \
    -e MODEL_CONFIG


# Save reference checkpoint if SAVE_REFERENCE_CKPT is set
if [ "$SAVE_REFERENCE_CKPT" = "1" ]; then
    echo "Saving reference checkpoint"
    srun /fsx/thangakr/save_rc.sh
fi

read_peak_memory

