#!/bin/bash
# Author: Bogomil Gospodinov
. scripts/slurm_init.sh

mkdir -p logs

slurm_active_jobs=$( squeue --format="%.18i %.9P %.20j" -u $(whoami) | awk '{ ORS=" " } NR > 1 { print $1 } $3 ~ /resume/ { sub(/resume_/, ""); print $3 }' )

declare -a jobids=()

shopt -s nullglob
for model_path in models/*/m*/[0-9]* ; do
	model_id=${model_path##*/};
	if [[ ! -f ${model_path%/*}/data/dev_prediction.${model_id} && -z "`grep $model_id <<< "$slurm_active_jobs"`" ]]; then
		jobids+=($model_id)
	fi
done

if [ ${#jobids[@]} -eq 0 ]; then
	echo "No jobs to rerun."
	exit 0
fi

echo Rerun ${jobids[*]} ?

read -p "Are you sure? (Yy/n) " -n 1 -r
echo # (optional) move to a new line
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
	[[ "$0" = "$BASH_SOURCE" ]] && exit 1 || return 1 # handle exits from shell or function but don't exit interactive shell
fi

for jobid in "${jobids[@]}" ; do
	echo Relaunching\ jobid=$jobid

	log_file=$(find logs/ -path \*${jobid}.log -print -quit)
	log_file=${log_file:="logs/%x.%j.log"}
	echo $log_file

	SLURM_ORIGINAL_JOB_ID=$jobid sbatch \
		--output=${log_file} --error=${log_file} \
		--open-mode=append \
		--mail-type=END,FAIL --mail-user="$(whoami)@sms.ed.ac.uk" \
		--partition=Standard,Short \
		--job-name="resume_${jobid}" scripts/train.sh

done
