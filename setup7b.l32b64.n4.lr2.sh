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

export MODEL_ARCH="fuji-7B-v2"
export N_LAYERS=32
export N_GBS=64
export N_ACCUMULATION=1
export OPTIMIZER_LR_BASE=7.5
export OPTIMIZER_LR_EXP=-5
export OPTIMIZER_WD=0.001
export N_EXPECTED_NODES=4 #For checks in main script to avoid mixing up configs

echo "GPU RUN CONFIG : MODEL=${MODEL_ARCH} L=${N_LAYERS} GBS=${N_GBS} ACC=${N_ACCUMULATION} OPTIMIZER_LR_BASE=${LR_BASE} OPTIMIZER_LR_EXP=${LR_EXP} OPTIMIZER_WD=${WD} N=${N_EXPECTED_NODES}"  

