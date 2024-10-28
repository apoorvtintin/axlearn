#!/usr/bin/env bash
WORKHOME="/shared/${USER}"
WORKROOT="${WORKHOME}/setup/drop2p1/axlearn"
sudo dpkg -i ${WORKROOT}/aws-neuronx-dkms_2.x.3740.0_amd64.deb
sudo dpkg -i ${WORKROOT}/aws-neuronx-collectives-2.x.18916.0-41121280a.deb
sudo dpkg -i ${WORKROOT}/aws-neuronx-runtime-lib-2.x.17742.0-83ba134d4.deb


PY_ENV_PATH="${WORKHOME}/venv/70b_drop2/bin/activate"
source ${PY_VENV_PATH}

cd /axlearn


ARTIFACTS_PATH="${WORKROOT}/runs"
TIMESTAMP=$(date +"%y%m%d%H%M%S")
TEST_ARTIFACTS_PATH="${ARTIFACTS_PATH}/t_$1/${TIMESTAMP}"
mkdir -p "$TEST_ARTIFACTS_PATH"

#UNIFIED MODEL CONFIG (WIP)
#Control configs here and echo in logs to avoid doubts on the config used for each run
#Please add relevant configs as we go
#TODO : Convert these from env var to args for launch_trainer_main for upstreaming
export NEURON_TP_SIZE=64
export NEURON_VNC_SIZE=2
export NEURON_FSDP=0 #TODO: enable when ready
export NEURON_FFP32_GRAD=0
export NEURON_REPLICA_UBATCH=16

NEURON_DUMP_PATH=${TEST_ARTIFACTS_PATH}/neuron_dump
HLO_DUMP_PATH=${TEST_ARTIFACTS_PATH}/hlo_dump
export XLA_FLAGS="--xla_dump_hlo_as_text --xla_disable_hlo_passes=aws_neuron_flip_all_gather_dot --xla_dump_hlo_as_proto --xla_dump_to=${HLO_DUMP_PATH} --xla_dump_hlo_pass_re='.*'"
#XLA_FLAGS="--xla_dump_hlo_snapshots ${XLA_FLAGS}"

# Neuron compiler flags
export NEURON_CC_FLAGS="--framework=XLA"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --target=trn2 --distribution-strategy=llm-training"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --internal-num-neuroncores-per-sengine=${NEURON_VNC_SIZE} --internal-hlo2tensorizer-options='--verify-hlo'"
export NEURON_RT_VIRTUAL_CORE_SIZE="${NEURON_VNC_SIZE}"
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

# Neuron runtime flags
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=1
export NEURON_RT_IO_RING_CACHE_SIZE=0
export NEURON_RT_ENABLE_MEMORY_METRICS=0
export OFI_NCCL_MR_CACHE_DISABLE=1

OUTPUT_DIR="${TEST_ARTIFACTS_PATH}/axlearn_out"
mkdir -p ${OUTPUT_DIR}
DATA_DIR="gs://axlearn-public/tensorflow_datasets"

echo "NEURON_RUN_INFO DIRECTORY: ${TEST_ARTIFACTS_PATH}" #Run artifacts are here
echo "NEURON_RUN_INFO CONFIG: TP=${NEURON_TP_SIZE} VNC=${NEURON_VNC_SIZE} FSDP=${NEURON_FSDP} FP32_GRAD=${NEURON_FFP32_GRAD} REPLICA_UBATCH=${NEURON_REPLICA_UBATCH}"


# Run the training script
python -m axlearn.common.launch_trainer_main \
    --module=text.gpt.c4_trainer --config=fuji-70B-v2 \
    --trainer_dir=$OUTPUT_DIR --data_dir=$DATA_DIR \
    --jax_backend=neuron --mesh_selector=trn2 
