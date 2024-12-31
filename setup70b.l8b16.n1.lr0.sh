#!/bin/bash

export MODEL_ARCH="fuji-70B-v2"
export N_LAYERS=8
export N_GBS=16
export N_ACCUMULATION=1
export OPTIMIZER_LR_BASE=1.5
export OPTIMIZER_LR_EXP=-5
export OPTIMIZER_WD=0.000006
export N_EXPECTED_NODES=1 #For checks in main script to avoid mixing up configs
export MESH_SELECTOR="gpu-70b-small-mesh"

