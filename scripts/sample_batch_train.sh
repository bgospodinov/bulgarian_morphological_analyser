#!/bin/bash
. slurm_init.sh

mkdir -p logs

for run in {1..3} ; do
	echo Run $run
	for attr in 10 15 20 25 ; do
		echo Launching\ attribute=$attr
		context_size=$attr sbatch --output="logs/train-%j.out" train.sh
	done
done
