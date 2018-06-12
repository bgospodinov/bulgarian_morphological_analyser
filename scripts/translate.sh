#!/bin/bash
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --gres-flags=enforce-binding
#SBATCH --gres=gpu:1
#SBATCH --mem=8000  # memory in Mb

[ -z "$model_dir" ] && echo "Export model_dir." && exit

# if slurm job id is unavailable use datetime for logging purposes
datetime_name=`date '+%Y-%m-%d_%H:%M:%S'`

set -x

SLURM_JOB_ID=${SLURM_JOB_ID:=$datetime_name}

# path to project root
base_dir=${base_dir:=..}

# path to nematus (relative to root dir project)
nematus=${nematus:=nematus}

model_dir=${model_dir}
input_path=${input_path:=${model_dir}/data/dev_source}
output_path=${output_path:=${model_dir}/data/dev_hypothesis}

set +x

currentdir=`pwd`
cd $base_dir

echo Translating

python ${nematus}/nematus/translate.py \
-m ${model_dir}/model.npz \
-i ${input_path} \
-o ${output_path} \
-k 12 -n -p 1 \
&> ${model_dir}/translate-${SLURM_JOB_ID}.out

cd $currentdir