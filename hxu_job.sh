#!/bin/bash
echo "Hello from ${JOB_ID}. "

# name
#$ -N TrainDecagon

# real memory
#$ -l rmem=16G

# core
#$ -pe openmp 1

# time
#$ -l h_rt=8:00:00

# mail
#$ -M hxu31@sheffield.ac.uk
# Email notifications if the job aborts
#$ -m a

# Setup env
module load apps/python/conda
source activate hxu-gcn

# run
python /home/acq18hx/decagon/main_sample.py