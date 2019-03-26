#!/bin/bash
echo "Hello from ${JOB_ID}. "

# name
#$ -N TrainDecagon

# real memory
#$ -l rmem=8G

# core
#$ -pe openmp 1

# time
#$ -l h_rt=24:00:00

# mail
#$ -M hxu31@sheffield.ac.uk

# Setup env
module load apps/python/conda
source activate hxu-sharc

# run
python /home/acq18hx/decagon/main.py