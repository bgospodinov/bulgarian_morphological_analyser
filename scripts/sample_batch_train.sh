#!/bin/bash
. slurm_init.sh

mkdir -p logs

for cs in 10 15 20 25 ; do
	echo Launching\ context_size=$cs
	context_size=$cs sbatch --output="logs/train-%j.out" train.sh
done
