#!/usr/bin/bash

set -e

# HOW TO RUN
# 1. Activate the (base) conda environment
# 2. Set OFI_PATH to your OFI installation
# 3  Optionally set JOB_ID, OUTROOT and OFI_PATH
#    - JOB_ID overwrites SLURM_JOB_ID to resume a CKPT from prior SLURM JOB_ID
# 4. Command line : run_trainer.sh env_name testname test_setup
#
CONDA_ENV_NAME=$1 #source your base conda env first
TESTNAME=$2
TEST_SETUP=$3
: ${OUTROOT:=${PWD}}
: ${JOB_ID:=${SLURM_JOB_ID}} 
#
# See OFI installation instructions here 
# https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start-nccl.html#nccl-start-base-plugin
OFI_DEFAULT_PATH="${HOME}/EFA/aws-ofi-nccl-aws/"
: ${OFI_PATH:=${OFI_DEFAULT_PATH}}

#Restores ckpt if TEST_ARTIFACT_PATH points at a prior TEST_ARTIFACT_PATH with a checkpoint
ARTIFACTS_PATH="${OUTROOT}/runs/artifacts"
TEST_ARTIFACTS_PATH="${ARTIFACTS_PATH}/$CONDA_ENV_NAME/t_${TESTNAME}/${JOB_ID}"
mkdir -p "$TEST_ARTIFACTS_PATH"
echo "RUN CONFIG : DIR ${TEST_ARTIFACTS_PATH} ${SLURM_JOB_ID} ${JOB_ID}"

NEURON_DUMP_PATH="${TEST_ARTIFACTS_PATH}/neuron_dump"
HLO_DUMP_PATH="${TEST_ARTIFACTS_PATH}/hlo_dump"
AXLEARN_PATH="${TEST_ARTIFACTS_PATH}/axlearn_out"
mkdir -p "$NEURON_DUMP_PATH"
mkdir -p "$HLO_DUMP_PATH"
mkdir -p "$AXLEARN_PATH"


#Debug Options
#export NCCL_DEBUG=INFO / VERSION
#export NCCL_DEBUG_SUBSYS=COLL


# Source CONDA environment
# Job must be launched from (base) conda environment 
which conda
source activate base
conda activate ${CONDA_ENV_NAME}


#Setup Run env vars
which python
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
num_nodes=${SLURM_NNODES}
MASTER_ADDR=$(echo "$nodes" | head -n 1)
MASTER_PORT=41008
JAX_COORDINATOR_PORT=41001
export NEURON_RT_ROOT_COMM_ID="${MASTER_ADDR}:${MASTER_PORT}"
export NEURON_PJRT_PROCESS_INDEX=${SLURM_NODEID}
export LD_LIBRARY_PATH="/opt/amazon/efa/lib/"
export LD_LIBRARY_PATH="$OFI_PATH:$LD_LIBRARY_PATH"
export FI_LOG_LEVEL="warn"
export FI_EFA_USE_DEVICE_RDMA="1"
export FI_PROVIDER="efa"
export FI_EFA_FORK_SAFE=1

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
# export JAX_PLATFORMS=cpu

#Perf Tuning Guideline here : https://github.com/NVIDIA/JAX-Toolbox/blob/main/rosetta/docs/PGLE.md
export XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=${HLO_DUMP_PATH} --xla_dump_hlo_pass_re='.*' --xla_dump_hlo_as_proto"
if [[ $GPU_RUN_TYPE == "profile" ]]; then
   echo "Executing GPU Profile Run"
   XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=false --xla_gpu_disable_async_collectives=allreduce,allgather,reducescatter,collectivebroadcast,alltoall,collectivepermute ${XLA_FLAGS}"
elif [[ $GPU_RUN_TYPE == "perf" ]]; then
   echo "Executing Profile Guided GPU Performance Run"
   if [[ -z "${GPU_PROFILE}" ]]; then
       echo "ERROR : Can not run GPU Performance Run without a profile"
       exit 1
   fi
   XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true
     --xla_gpu_enable_triton_gemm=false
     --xla_gpu_graph_level=0
     --xla_gpu_all_reduce_combine_threshold_bytes=1073741824
     --xla_gpu_all_gather_combine_threshold_bytes=1073741824
     --xla_gpu_reduce_scatter_combine_threshold_bytes=1073741824
     --xla_gpu_enable_pipelined_all_gather=true
     --xla_gpu_enable_pipelined_reduce_scatter=true
     --xla_gpu_enable_pipelined_all_reduce=true
     --xla_gpu_enable_while_loop_double_buffering=true
     --xla_gpu_enable_all_gather_combine_by_dim=false
     --xla_gpu_enable_reduce_scatter_combine_by_dim=false
     --xla_disable_hlo_passes=rematerialization
     --xla_gpu_pgle_profile_file_or_directory_path=${GPU_PROFILE}
     ${XLA_FLAGS}" 
else
   # Perf run with no profile
   echo "Executing GPU Run Default Settings"
   #XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_while_loop_double_buffering=true --xla_gpu_enable_pipelined_all_gather=true --xla_gpu_enable_pipelined_reduce_scatter=true --xla_gpu_enable_pipelined_all_reduce=true --xla_gpu_multi_streamed_windowed_einsum=true --xla_gpu_enable_custom_fusions=true --xla_gpu_enable_address_computation_fusion=true ${XLA_FLAGS}" 
   # NOTE: The perf options above results in bad convergence. Disable it and use the profile run options instead
   #  - Additionally, this perf flag fails --xla_gpu_enable_address_computation_fusion=true
   XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=false --xla_gpu_disable_async_collectives=allreduce,allgather,reducescatter,collectivebroadcast,alltoall,collectivepermute ${XLA_FLAGS}"
fi


#Setup Datasets and test
DATA_DIR="gs://axlearn-public/tensorflow_datasets"
source ${TEST_SETUP}
if [ "${N_EXPECTED_NODES}" -ne "${num_nodes}" ]; then
    echo "ERROR : ${TEST_SETUP} for ${N_EXPECTED_NODES} was launched with ${num_nodes}"
    exit 1
fi

# Run the training script
echo "distributed_coordinator " $NEURON_RT_ROOT_COMM_ID
echo "num_processes " $num_nodes
echo "process_id " $NEURON_PJRT_PROCESS_INDEX
hostname
# key config for convenience only
echo "RUN CONFIG : MODEL=${MODEL_ARCH} L=${N_LAYERS} GBS=${N_GBS} ACC=${N_ACCUMULATION} OPTIMIZER_LR_BASE=${OPTIMIZER_LR_BASE} OPTIMIZER_LR_EXP=${OPTIMIZER_LR_EXP} OPTIMIZER_WD=${OPTIMIZER_WD} N=${N_EXPECTED_NODES} MESH=${MESH_SELECTOR}"
printenv  #Complete final env just before launch
python -u -m axlearn.common.launch_trainer_main \
    --module=text.gpt.c4_trainer --config=$MODEL_ARCH \
    --trainer_dir=$AXLEARN_PATH --data_dir=$DATA_DIR \
    --jax_backend=gpu --distributed_coordinator=$NEURON_RT_ROOT_COMM_ID \
    --num_processes=$num_nodes \
    --process_id=$NEURON_PJRT_PROCESS_INDEX \
    --mesh_selector=$MESH_SELECTOR

