# Install a different tools package if requested
TOOLS_DEB=$(find /overrides/tools/ -name "*.deb" -type f -print 2>/dev/null | head -n 1)
if [ -n "$TOOLS_DEB" ]; then
    echo "Installing tools"
    dpkg -i --force-all "$TOOLS_DEB"
fi

# Install a different runtime if requested
RUNTIME_DEB=$(find /overrides/runtime/ -name "*.deb" -type f -print 2>/dev/null | head -n 1)
if [ -n "$RUNTIME_DEB" ]; then
    echo "Installing runtime"
    dpkg -i --force-all "$RUNTIME_DEB"
fi

# Install a different cclib if requested
CCLIB_DEB=$(find /overrides/cclib/ -name "*.deb" -type f -print 2>/dev/null | head -n 1)
if [ -n "$CCLIB_DEB" ]; then
    echo "Installing cclib"
    dpkg -i --force-all "$CCLIB_DEB"
fi

# Install a different pjrt plugin if requested 
PJRT_WHL=$(find /overrides/pjrt/ -name "*.whl" -type f -print 2>/dev/null | head -n 1)
if [ -n "$PJRT_WHL" ]; then
      echo "Installing PJRT"
          pip install --force-reinstall --no-deps --extra-index-url=https://pip.repos.neuron.amazonaws.com "$PJRT_WHL"
fi

# Install a different compiler binary if requested
COMPILER_WHL=$(find /overrides/compiler/ -name "*.whl" -type f -print 2>/dev/null | head -n 1)
if [ -n "$COMPILER_WHL" ]; then
      echo "Installing compiler"
          pip install --force-reinstall "$COMPILER_WHL"
fi

# Install axlearn from a different path if requested
# Check if both AXLEARN_CLONE_URL and AXLEARN_CHECKOUT are set
if [ -n "$AXLEARN_CLONE_URL" ] && [ -n "$AXLEARN_CHECKOUT" ]; then
    # Remove the existing /axlearn directory
    rm -rf /axlearn
    # Clone the git repo
    echo "Cloning axlearn from $AXLEARN_CLONE_URL"
    git clone "$AXLEARN_CLONE_URL" /axlearn
    # Change to the /axlearn directory
    pushd /axlearn > /dev/null || exit 1
    # Checkout the specified branch or commit
    echo "Checking out $AXLEARN_CHECKOUT"
    git checkout "$AXLEARN_CHECKOUT"
    # Install axlearn in editable mode
    pip install --force-reinstall -e .
    # Return to the original directory
    popd > /dev/null
elif [ -n "$AXLEARN_CLONE_URL" ] || [ -n "$AXLEARN_CHECKOUT" ]; then
    echo "Error: Both AXLEARN_CLONE_URL and AXLEARN_CHECKOUT must be set."
    exit 1
fi

# Reinstall axlearn if needed. This happens when user asks to use axlearn from a local folder using AXLEARN_PATH
if [ -n "$REINSTALL_AXLEARN" ]; then
    # Change to the /axlearn directory and remember the current directory
    pushd /axlearn > /dev/null || exit 1

    # Print installation message
    echo "Installing axlearn from $AXLEARN_PATH"
    # Check if it's a git repository and print status
    if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        echo "Git status:"
        git status
    else
        echo "Not a git repository"
    fi

    # Install axlearn in editable mode
    pip install --force-reinstall -e .
    # Return to the original directory
    popd > /dev/null
fi

# Update path
export PATH=/opt/aws/neuron/bin:${PATH}

# FSDP
if [ -n "${USE_FSDP}" ] && [ "${USE_FSDP}" -ne 0 ]; then
    export NEURON_FSDP=1
fi

export VNC=${LNC}
export TRN=trn2

source /axltest/axltest/flags.sh

OUTPUT_DIR="/mnt/artifacts/axlearn_out"
mkdir -p ${OUTPUT_DIR}

echo "Packages:"
apt list | grep neuron
pip list | grep neuron

# Environment variables:
set

DATA_DIR="gs://axlearn-public/tensorflow_datasets"
# Set default model config if not specified
MODEL_CONFIG=${MODEL_CONFIG:-fuji-70B-v2-flash}

python -m axlearn.common.launch_trainer_main \
    --module=axltest.c4_trainer \
    --config=${MODEL_CONFIG} \
    --trainer_dir=${OUTPUT_DIR} \
    --data_dir=$(if [ -z "$USE_FAKE_DATA" ]; then echo "$DATA_DIR"; else echo "FAKE"; fi) \
    --jax_backend=neuron \
    --mesh_selector=neuron-trn2.48xlarge-64 \
    --distributed_coordinator=$NEURON_PJRT_COORDINATOR_COMM_ID --num_processes=$NEURON_N_PROC \
    --process_id=$NEURON_PJRT_PROCESS_INDEX

if [ "${PROFILE_AFTER_RUN:-0}" -eq 1 ]; then
  most_recent_dir="$(find "$NEURON_DUMP_PATH" -maxdepth 1 -type d -name 'pid*-program*' -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
  if [ -n "$most_recent_dir" ] && [ -f "$most_recent_dir/file.neff" ]; then
    cd "$most_recent_dir" || exit 1
    echo "processing neuron dump directory: $(pwd)"

    export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=0
    export NEURON_RT_NUM_CORES=$((128 / ${LNC}))

    ranks_per_node=$((128 / ${VNC}))
    ranks_in_cluster=$((ranks_per_node * NEURON_N_PROC))
    neuron-profile capture \
      -n file.neff \
      --num-exec 3 \
      --collectives-worker-count ${ranks_in_cluster} \
      --collectives-worker-start-id $((NEURON_PJRT_PROCESS_INDEX * ranks_per_node)) \
      --collectives-workers-per-node ${ranks_per_node} \
      --collectives-profile-id 0 \
      -s profile.ntff

    ntff_file=$(ls *.ntff 2>/dev/null)
    if [ -n "$ntff_file" ]; then
      mv "$ntff_file" profile.ntff

      # penguin artifacts for profiler
      if [ -n "$INTERNAL_DEBUG_MODE" ] && [[ "${INTERNAL_DEBUG_MODE}" =~ (penguin|all) ]]; then
          tar -cvf penguin-text.tar penguin-sg*
          tar tf penguin-text.tar | grep "penguin.txt"
      fi
    else
      echo "No profile.ntff file found"
    fi
  else
    echo "Neuron dump folder or file.neff not found"
    exit 1
  fi
fi
