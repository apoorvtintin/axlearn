#!/bin/bash

export MODEL_ARCH="fuji-7B-v2"
export N_LAYERS=32
export N_GBS=64
export N_ACCUMULATION=1
export OPTIMIZER_LR_BASE=1.5
export OPTIMIZER_LR_EXP=-5
export OPTIMIZER_WD=0.000006
export N_EXPECTED_NODES=4 #For checks in main script to avoid mixing up configs
export MESH_SELECTOR="gpu-7b-mesh"

