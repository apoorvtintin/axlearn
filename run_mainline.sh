#!/usr/bin/env bash

# Neuron env vars for distributed training based on SLURM
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
num_nodes=$(echo "$nodes" | wc -l)
devices_per_node=64
MASTER_ADDR=$(echo "$nodes" | head -n 1)
MASTER_PORT=41000
JAX_COORDINATOR_PORT=41001
export NEURON_RT_ROOT_COMM_ID="${MASTER_ADDR}:${MASTER_PORT}"
export NEURON_PJRT_PROCESSES_NUM_DEVICES=$(printf '%s,' $(seq 1 $num_nodes | xargs -I {} echo $devices_per_node) | sed 's/,$//')
export NEURON_PJRT_PROCESS_INDEX=$SLURM_NODEID
export LD_LIBRARY_PATH="/opt/amazon/efa/lib/"
export FI_LOG_LEVEL="warn"
export FI_EFA_USE_DEVICE_RDMA="1"
export FI_PROVIDER="efa"
export FI_EFA_FORK_SAFE=1

sudo apt-get -f install -y
sudo apt-get install -y google-perftools

sudo dpkg -i /fsx/apoorvgu/aws-neuronx-runtime-lib-2.x.19993.0-1bf746e12.deb
sudo dpkg -i /fsx/apoorvgu/aws-neuronx-collectives-2.x.21370.0-8cbb4877b.deb

if ! apt list 2>/dev/null | grep -q "^aws-neuronx-dkms/now 2.x.4125.0 amd64 \[installed,local\]"; then sudo dpkg -i --force-all /fsx/thangakr/binaries/aws-neuronx-dkms_2.x.4125.0_amd64.deb; fi
CHECK_STATUS=$?
if [ $CHECK_STATUS -ne 0 ]; then
    echo "Driver version check failed! Terminating job."
    exit 1
fi

hostname


JOB_ID=${SLURM_JOB_ID}
ARTIFACTS_PATH="/fsx/apoorvgu/upstream_final/final_artifacts/"
# TIMESTAMP=$(date +"%y%m%d%H%M%S")
TEST_ARTIFACTS_PATH="${ARTIFACTS_PATH}/${JOB_ID}"
mkdir -p "$TEST_ARTIFACTS_PATH"

NEURON_DUMP_PATH=${TEST_ARTIFACTS_PATH}/neuron_dump
HLO_DUMP_PATH=${TEST_ARTIFACTS_PATH}/hlo_dump

export NEURON_RT_DBG_CC_DMA_PACKET_SIZE=4096 && export NEURON_RT_DBG_DMA_PACKETIZATION_SIZE=104857
export NEURON_FSDP_NUM_LAYER_EARLY_AG_SHIFT=1
export NEURON_FSDP_NUM_LAYER_LATE_RS_SHIFT=2

export NEURON_ENABLE_INT_MATMUL_DOWNCAST=1


# Neuron runtime flags
# export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=1
export NEURON_RT_IO_RING_CACHE_SIZE=0
export NEURON_RT_ENABLE_MEMORY_METRICS=0
export NEURON_RT_VIRTUAL_CORE_SIZE=2
export NEURON_RT_RESET_CORES=1
export NEURON_RT_LOG_LEVEL="WARNING"
export NEURON_RUN_TRIVIAL_COMPUTATION_ON_CPU=1

export NEURON_RT_ENABLE_INTERNODE_EXECUTION_BARRIER=1

# export NEURON_ALL_REDUCE_UPCASTER=1
export NEURON_FSDP_NUM_LAYER_COALESCE=-1

# Neuron collectives flag
export FI_LOG_LEVEL="warn"
export OFI_NCCL_PROTOCOL=RDMA
export LD_LIBRARY_PATH="/opt/amazon/efa/lib/"
export FI_EFA_USE_DEVICE_RDMA="1"
export FI_PROVIDER="efa"
export FI_EFA_FORK_SAFE=1
export OFI_NCCL_MR_CACHE_DISABLE=1

# Neuron compiler flags
export NEURON_CC_FLAGS="--framework=XLA"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --internal-max-instruction-limit=20000000"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --target=trn2"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --internal-num-neuroncores-per-sengine=2"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --model-type transformer"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --no-internal-hlo-remat"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --enable-mixed-precision-accumulation"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} -O1"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --tensorizer-options='--enable-hoist-fsdp-collectives'"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --internal-hlo2tensorizer-options='--remat-rope'"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --dump=${NEURON_DUMP_PATH}"

# export JAX_COMPILATION_CACHE_DIR="/fsx/apoorvgu/upstream_final/cc_cache/"
# mkdir -p ${JAX_COMPILATION_CACHE_DIR}

# export NEURON_FSDP=1
export DATA_SEED=1
export LNC=2
# export NEURON_ALL_REDUCE_UPCASTER=1

# export TF_CPP_MIN_LOG_LEVEL=0 # Enable SPMD verbose logging - 0 means most verbose
# export TF_CPP_MAX_VLOG_LEVEL=0 # Needs above flag for logging but goes in reverse. 0 means no log
# export TF_CPP_MIN_LOG_LEVEL=0
# export TF_CPP_MAX_VLOG_LEVEL=0

# conda
# eval "$(/fsx/apoorvgu/conda/bin/conda shell.bash hook)"
# conda activate py310
deactivate
source /fsx/apoorvgu/envs/new_env/bin/activate

echo "Listing apt dependencies"
apt list --installed | grep neuron
echo "Listing pip dependencies"
pip list | grep neuron
echo "Done listing dependencies"

printenv | grep NEURON
printenv | grep XLA

which python
LIBTCMALLOC=$(find /usr/lib/x86_64-linux-gnu -name "libtcmalloc.so.*" | sort -V | tail -n 1)
 
if [ -n "$LIBTCMALLOC" ]; then
    # Create a symbolic link to the found libtcmalloc version
    sudo ln -sf "$LIBTCMALLOC" /usr/lib/libtcmalloc.so
    echo "Symbolic link created: /usr/lib/libtcmalloc.so -> $LIBTCMALLOC"
 
    # Export LD_PRELOAD
    export LD_PRELOAD=/usr/lib/libtcmalloc.so
    echo "LD_PRELOAD set to: $LD_PRELOAD"
else
    echo "Error: libtcmalloc.so not found"
    exit 1
fi

mkdir -p ${OUTPUT_DIR}
DATA_DIR="gs://axlearn-public/tensorflow_datasets"

# Check if the CPU argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 cpu=<1|0>"
    exit 1
fi

# Extract the value after "cpu="
CPU_VALUE=${1#*=}

if [ "$CPU_VALUE" = "1" ]; then
    OUTPUT_DIR="/fsx/apoorvgu/upstream_final/final_artifacts/axlearn_cpu"
    export XLA_FLAGS="--xla_force_host_platform_device_count=64 --xla_dump_hlo_as_text --xla_dump_to=${HLO_DUMP_PATH} --xla_dump_hlo_pass_re='.*'"
    export JAX_PLATFORMS=cpu
    echo "Running CPU"
    python -m axlearn.common.launch_trainer_main \
        --module=text.gpt.c4_trainer --config=fuji-70B-v2 \
        --trainer_dir=$OUTPUT_DIR --data_dir=$DATA_DIR \
        --jax_backend=cpu --mesh_selector=neuron-trn2.48xlarge-64 \
        --distributed_coordinator=$MASTER_ADDR:$JAX_COORDINATOR_PORT --num_processes=$num_nodes \
        --process_id=$NEURON_PJRT_PROCESS_INDEX

elif [ "$CPU_VALUE" = "0" ]; then
    echo "Running Neuron"
    OUTPUT_DIR="/fsx/apoorvgu/upstream_final/final_artifacts/axlearn_trn"
    export XLA_FLAGS="--xla_dump_hlo_as_text --xla_disable_hlo_passes=aws_neuron_flip_all_gather_dot,neuron-hierarchical-collectives,neuron_all_gather_duplicate_remover,neuron-token-threading,aws_neuron_dynamic_slice_reshape_canonicalizer --xla_dump_to=${HLO_DUMP_PATH} --xla_dump_hlo_pass_re='.*'"
    python -m axlearn.common.launch_trainer_main \
        --module=text.gpt.c4_trainer --config=fuji-70B-v2 \
        --trainer_dir=$OUTPUT_DIR --data_dir=$DATA_DIR \
        --jax_backend=neuron --mesh_selector=neuron-trn2.48xlarge-64 \
        --distributed_coordinator=$MASTER_ADDR:$JAX_COORDINATOR_PORT --num_processes=$num_nodes \
        --process_id=$NEURON_PJRT_PROCESS_INDEX
else
    echo "Invalid argument. Please use 'cpu=1' to enable CPU mode or 'cpu=0' to disable."
    exit 1
fi