#!/bin/bash

#SBATCH --job-name=gpu_decagon   # job name
#SBATCH --output=gpu.out # output log file
#SBATCH --error=gpu.err  # error file
#SBATCH --time=36:00:00  # 1 hour of wall time
#SBATCH --nodes=1        # 1 GPU node
#SBATCH --partition=gpu2 # GPU2 partition
#SBATCH --ntasks=1       # 1 CPU core to drive GPU
#SBATCH --gres=gpu:1     # Request 1 GPU

# Load all required modules below. As an example we load cuda/9.1
module load cuda/9.1

# Add lines here to run your GPU-based computations.
python decagon/hhhhh/main.py
