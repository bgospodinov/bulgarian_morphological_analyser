#!/bin/bash
# Author: Bogomil Gospodinov
. scripts/slurm_init.sh

mkdir -p logs

declare -a jobids=(128000)

for jobid in "${jobids[@]}" ; do
	echo Relaunching\ jobid=$jobid

	SLURM_ORIGINAL_JOB_ID=$jobid sbatch \
		--output="logs/%x.%j.log" --error="logs/%x.%j.log" \
		--mail-type=END,FAIL --mail-user="$(whoami)@sms.ed.ac.uk" \
		--job-name="resume_${jobid}" scripts/train.sh

done
