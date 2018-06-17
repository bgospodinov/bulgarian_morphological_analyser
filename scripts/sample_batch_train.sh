#!/bin/bash
. slurm_init.sh

mkdir -p logs

runs=3

declare -a attrs=(100 500 750 1000 1200)

# 1 if the corresponding job should go to the LongJobs partition, 0 otherwise
declare -a lj_pred=(0 0 0 1 1)

for (( run=1; run<=$runs; run++ ))
do
	echo Run $run
	iter=0
	for attr in "${attrs[@]}" ; do
		echo Launching\ attribute=$attr
		context_size=$attr sbatch --output="logs/train-%j.out" --job-name="${run}-${attr}" $( (( ${lj_pred[$iter]} == 1 )) && printf %s '--partition=LongJobs' ) train.sh
		((iter++))
	done
done
