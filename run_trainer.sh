#!/usr/bin/env bash
sudo dpkg -i /home/apoorvgu/aws-neuronx-collectives-2.x.x.x-a311abe53.deb
sudo dpkg -i /home/apoorvgu/aws-neuronx-runtime-lib-2.x.x.x-a7a599f60.deb
sudo dpkg -i /home/apoorvgu/aws-neuronx-dkms_2.x.3951.0_amd64.deb
sudo dpkg -i /shared/huilgolr/env_builders/tot-binaries/aws-neuronx-tools-2.0.8969.0.deb
PY_VENV_PATH="/shared/apoorvgu/py310/bin/activate"
source ${PY_VENV_PATH}

cd /axlearn

ARTIFACTS_PATH="/shared/apoorvgu/artifacts"
TIMESTAMP=$(date +"%y%m%d%H%M%S")
TEST_ARTIFACTS_PATH="${ARTIFACTS_PATH}/${TIMESTAMP}"
mkdir -p "$TEST_ARTIFACTS_PATH"

NEURON_DUMP_PATH=${TEST_ARTIFACTS_PATH}/neuron_dump
HLO_DUMP_PATH=${TEST_ARTIFACTS_PATH}/hlo_dump
export XLA_FLAGS="--xla_dump_hlo_as_text --xla_disable_hlo_passes=aws_neuron_flip_all_gather_dot,neuron-hierarchical-collectives --xla_dump_hlo_as_proto --xla_dump_to=${HLO_DUMP_PATH} --xla_dump_hlo_pass_re='.*'"

# export TF_CPP_MIN_LOG_LEVEL=0 # Enable SPMD verbose logging - 0 means most verbose
# export TF_CPP_MAX_VLOG_LEVEL=0 # Needs above flag for logging but goes in reverse. 0 means no log
# export TF_CPP_MIN_LOG_LEVEL=0
# export TF_CPP_MAX_VLOG_LEVEL=0

# export TF_CPP_VMODULE='neuron_token_threading=5,neuron_fsdp_all_gather_split=5,neuron_hierarchical_collectives=5,neuron_all_gather_combiner=5,neuron_reduce_scatter_combiner=5'
export LNC=2
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

# Neuron compiler flags
export NEURON_CC_FLAGS="--framework=XLA"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --target=trn2" # --distribution-strategy=llm-training"
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
export NEURON_FSDP_NUM_LAYER_COALESCE=1
export DISABLE_REWRITE_REPLICA_GROUP_HLO_PASS=1
export NEURON_RT_ENABLE_INTERNODE_EXECUTION_BARRIER=1
export NEURON_RT_DBG_DISABLE_POD=1
export NEURON_RT_EXEC_TIMEOUT=180

export REMAT_STYLE="experiment"
export TP_DEGREE=4
export N_LAYERS=6
export ENABLE_NEW_UNSHARDED_ATTN_KERNEL=1
export MAX_TRAIN_STEPS=100

# Neuron runtime flags
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=1
export NEURON_RT_IO_RING_CACHE_SIZE=0
export NEURON_RT_ENABLE_MEMORY_METRICS=0

# Function to write zero to all peak files
write_zero_to_peaks() {
    local base_path="/sys/devices/virtual/neuron_device/neuron0/neuron_core0/stats/memory_usage/device_mem"
    local categories=("collectives" "constants" "dma_rings" "driver_memory" "model_code" "model_shared_scratchpad" "nonshared_scratchpad" "notifications" "runtime_memory" "tensors" "uncategorized")

    for category in "${categories[@]}"; do
        sudo bash -c "echo 0 > ${base_path}/${category}/peak"
    done
    sudo bash -c "echo 0 > ${base_path}/peak"
}

# Function to read and print summary of peak memory
read_peak_memory() {
    local base_path="/sys/devices/virtual/neuron_device/neuron0/neuron_core0/stats/memory_usage/device_mem"
    local categories=("collectives" "constants" "dma_rings" "driver_memory" "model_code" "model_shared_scratchpad" "nonshared_scratchpad" "notifications" "runtime_memory" "tensors" "uncategorized")

    # Function to convert bytes to GB with 3 decimal places
    bytes_to_gb() {
        echo "scale=3; $1 / 1024 / 1024 / 1024" | bc
    }

    echo "Peak Memory Summary:"
    for category in "${categories[@]}"; do
        local peak=$(sudo cat "${base_path}/${category}/peak")
        local peak_gb=$(bytes_to_gb $peak)
        printf "%s: %s GB\n" "${category}" "${peak_gb}"
    done

    local total_peak=$(sudo cat "${base_path}/peak")
    local total_peak_gb=$(bytes_to_gb $total_peak)
    printf "Total Peak: %s GB\n" "${total_peak_gb}"
}

write_zero_to_peaks

OUTPUT_DIR="${TEST_ARTIFACTS_PATH}/axlearn_out"
mkdir -p ${OUTPUT_DIR}
DATA_DIR="gs://axlearn-public/tensorflow_datasets"
# Run the training script
python -m axlearn.common.launch_trainer_main \
    --module=text.gpt.c4_trainer --config=fuji-70B-v2 \
    --trainer_dir=$OUTPUT_DIR --data_dir=$DATA_DIR \
    --jax_backend=neuron --mesh_selector=trn2 \
    --distributed_coordinator=$MASTER_ADDR:$JAX_COORDINATOR_PORT --num_processes=$num_nodes \
    --process_id=$NEURON_PJRT_PROCESS_INDEX

read_peak_memory
