#!/usr/bin/env bash
sudo dpkg -i /shared/huilgolr/env_builders/tot-binaries/aws-neuronx-runtime-lib-2.x.17887.0-62bff0cbe
sudo dpkg -i /shared/huilgolr/env_builders/tot-binaries/aws-neuronx-collectives-2.x.19099.0-8a7991b2a.deb
sudo dpkg -i /shared/huilgolr/env_builders/tot-binaries/aws-neuronx-dkms_2.x.3784.0_amd64
sudo dpkg -i /shared/huilgolr/env_builders/tot-binaries/aws-neuronx-tools-2.0.8969.0.deb
PY_VENV_PATH="/shared/apoorvgu/py310/bin/activate"
source ${PY_VENV_PATH}

which python

cd /axlearn

ARTIFACTS_PATH="/shared/apoorvgu/artifacts"
TIMESTAMP=$(date +"%y%m%d%H%M%S")
TEST_ARTIFACTS_PATH="${ARTIFACTS_PATH}/${TIMESTAMP}"
mkdir -p "$TEST_ARTIFACTS_PATH"

NEURON_DUMP_PATH=${TEST_ARTIFACTS_PATH}/neuron_dump
HLO_DUMP_PATH=${TEST_ARTIFACTS_PATH}/hlo_dump
export XLA_FLAGS="--xla_dump_hlo_as_text --xla_disable_hlo_passes=aws_neuron_flip_all_gather_dot --xla_dump_hlo_as_proto --xla_dump_to=${HLO_DUMP_PATH} --xla_dump_hlo_pass_re='.*'" # --xla_force_host_platform_device_count=16"


# Neuron compiler flags
export NEURON_CC_FLAGS="--framework=XLA"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --target=trn2 --distribution-strategy=llm-training"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --internal-num-neuroncores-per-sengine=2 --internal-hlo2tensorizer-options='--verify-hlo'"
export NEURON_RT_VIRTUAL_CORE_SIZE=2
export NEURON_RT_RESET_CORES=1
export NEURON_RT_LOG_LEVEL="WARNING"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --target=trn2"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --model-type transformer"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --no-internal-hlo-remat"

export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --enable-mixed-precision-accumulation"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} -O1"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --dump=${NEURON_DUMP_PATH}"

# Neuron PJRT flags
export NEURON_WHILE_LOOP_UNROLL=1
export NEURON_RUN_TRIVIAL_COMPUTATION_ON_CPU=1
export TRN2=1
export NEURON_FSDP=1

# Neuron runtime flags
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=1
export NEURON_RT_IO_RING_CACHE_SIZE=0
export NEURON_RT_ENABLE_MEMORY_METRICS=0
export OFI_NCCL_PROTOCOL=RDMA
export OFI_NCCL_MR_CACHE_DISABLE=1

# export JAX_PLATFORMS=cpu

OUTPUT_DIR="${TEST_ARTIFACTS_PATH}/axlearn_out"
mkdir -p ${OUTPUT_DIR}
DATA_DIR="gs://axlearn-public/tensorflow_datasets"
# Run the training script
python -m axlearn.common.launch_trainer_main \
    --module=text.gpt.c4_trainer --config=fuji-70B-v2 \
    --trainer_dir=$OUTPUT_DIR --data_dir=$DATA_DIR \
    --jax_backend=neuron --mesh_selector=trn2 

