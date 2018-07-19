#!/bin/bash
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --gres-flags=enforce-binding
#SBATCH --gres=gpu:1
#SBATCH --mem=8000  # memory in Mb

# Author: Bogomil Gospodinov
[ -z "$job_id" ] && echo "You must provide job_id of existing model." && exit

set -x

job_id=$job_id

# path to nematus (relative to root dir project)
nematus=${nematus:=nematus}

model_run_dir=$( find models/ -type d -path "*/$job_id" | head -n 1 )
model_dir=${model_run_dir%/*}
# NB dev_source below already contains the context tags on the left....
input_path=${input_path:=${model_dir}/data/dev_source}
output_path=${output_path:=${model_dir}/data/dev_hypothesis.ct.${job_id}}

set +x

echo Translating

python ${nematus}/nematus/translate_sentence.py \
-m ${model_run_dir}/model.npz \
-i ${input_path} \
-o ${output_path} \
-k 12 -n -p 1 -v