#!/bin/bash
#SBATCH --output=slurm-%x-%j.out
#SBATCH --exclusive
#SBATCH --nodes=2
#SBATCH --reservation=training

srun  --kill-on-bad-exit=1  run_trainer.sh "$@"
