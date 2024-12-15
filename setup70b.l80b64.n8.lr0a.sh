#!/bin/bash



#MODEL CONFIGa
#To avoid OOM Max value for N_ACCUMULATION are :
# - 7B  8L  = 
# - 7B  32L =
# - 70B 8L  =
# - 70B 80L = 
#Using the max value will get the best throughput.
#NOTE : This was measured at 4nodes. May be higher at higher node count
#
#Things that can be explored for more tuning
# Parallelism degrees, especially TP, PP, SP, DP degrees instead of FSDP
# Reprofiling and retuning CC Buffer sizing on optimal parallelism degrees

export MODEL_ARCH="fuji-70B-v2"
export N_LAYERS=80
export N_GBS=64
export N_ACCUMULATION=1
export OPTIMIZER_LR_BASE=3.75
export OPTIMIZER_LR_EXP=-6
export OPTIMIZER_WD=0.000006
export N_EXPECTED_NODES=8 #For checks in main script to avoid mixing up configs

echo "GPU RUN CONFIG : MODEL=${MODEL_ARCH} L=${N_LAYERS} GBS=${N_GBS} ACC=${N_ACCUMULATION} OPTIMIZER_LR_BASE=${OPTIMIZER_LR_BASE} OPTIMIZER_LR_EXP=${OPTIMIZER_LR_EXP} OPTIMIZER_WD=${OPTIMIZER_WD} N=${N_EXPECTED_NODES}"  

