#!/bin/bash
# Author: Bogomil Gospodinov
. scripts/slurm_init.sh

mkdir -p logs

runs=3

declare -a attrs=(100 300 500 750 1000)

# 1 if the corresponding job should go to the LongJobs partition, 0 otherwise
declare -a lj_pred=(0 0 0 0 0)

for (( run=1; run<=$runs; run++ ))
do
	echo Run $run
	iter=0
	for attr in "${attrs[@]}" ; do
		echo Launching\ attribute=$attr

		state_size=$attr seed=$run transform_mode=word_and_context sbatch \
			--output="logs/%x.%j.log" --error="logs/%x.%j.log" \
			--begin=now+0hour \
			--open-mode=append \
			--mail-type=END,FAIL --mail-user="$(whoami)@sms.ed.ac.uk" \
			--job-name="${run}_${attr}" --partition=$( (( ${lj_pred[$iter]} == 1 )) && printf %s 'LongJobs' || printf %s 'Standard,Short' ) scripts/train.sh

		((iter++))
	done
done
