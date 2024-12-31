#!/bin/bash

export MODEL_ARCH="fuji-70B-v2"
export N_LAYERS=80
export N_GBS=128
export N_ACCUMULATION=2
export OPTIMIZER_LR_BASE=7.5
export OPTIMIZER_LR_EXP=-6
export OPTIMIZER_WD=0.000006
export N_EXPECTED_NODES=8 #For checks in main script to avoid mixing up configs
export MESH_SELECTOR="gpu-70b-small-mesh"

