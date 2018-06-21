#!/bin/bash
# Author: Bogomil Gospodinov
. scripts/slurm_init.sh

mkdir -p logs

declare -a jobids=()

for jobid in "${jobids[@]}" ; do
	echo Translating dev set for model jobid=$jobid

	SLURM_ORIGINAL_JOB_ID=$jobid skip_resume_training=1 sbatch \
		--output="logs/%x.%j.log" --error="logs/%x.%j.log" \
		--mail-type=END,FAIL --mail-user="$(whoami)@sms.ed.ac.uk" \
		--job-name="resume_${jobid}" scripts/train.sh

done
