#!/bin/bash
. scripts/slurm_init.sh

set -x
model_dir=models/MorphoData-NewSplit_wchar_tchar_20u_cchar_n1_ct_cs1/m3_1_300_300_tanh_0.0_0.2_0.3_0.0_0.0_adadelta_1.0
job_ids=($model_dir/[0-9]*)
datetime_name=`date +%s`
SLURM_JOB_ID=${SLURM_JOB_ID:=$datetime_name}
set +x
set -e

for i in ${job_ids[@]}; do
	i=${i##*/}
	echo Translating $i
	job_id=$i srun scripts/translate_sentence.sh | tee logs/${SLURM_JOB_ID}.ct.log
	
	echo Postprocessing dev predictions
	/usr/bin/time -f %e python -m data.postprocess_nematus ${model_dir}/data/dev_hypothesis.ct.$i ${model_dir}/data/dev_source > ${model_dir}/data/dev_prediction.ct.$i

	echo Calculating score
	/usr/bin/time -f %e python -m analysis.score_prediction ${model_dir}/data/dev_prediction.ct.$i > ${model_dir}/data/dev_score.ct.$i

	echo Concatenating
	cat ${model_dir}/data/dev_score.ct.* > ${model_dir}/data/dev_scores.ct

	echo Averaging
	/usr/bin/time -f %e python -m analysis.average ${model_dir}/data/dev_scores.ct > ${model_dir}/data/dev_avg_score.ct

done
