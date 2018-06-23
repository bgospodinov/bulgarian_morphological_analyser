#!/bin/bash
# Author: Bogomil Gospodinov
. scripts/slurm_init.sh

mkdir -p logs

declare -a jobids=()

for jobid in "${jobids[@]}" ; do
	echo Translating dev set for model jobid=$jobid

	log_file=$(find logs/ -path \*${jobid}.log -print -quit)
	log_file=${log_file:="logs/%x.%j.log"}
	echo $log_file

	SLURM_ORIGINAL_JOB_ID=$jobid skip_resume_training=1 sbatch \
		--output=${log_file} --error=${log_file} \
		--open-mode=append \
		--partition=Standard,Short \
		--mail-type=END,FAIL --mail-user="$(whoami)@sms.ed.ac.uk" \
		--job-name="resume_${jobid}" scripts/train.sh

done
