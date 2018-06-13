#!/bin/bash
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --mem=8000  # memory in Mb

# path to original dataset (relative to root dir of project)
original_dataset=data/datasets/MorphoData-NewSplit

# if slurm job id is unavailable use datetime for logging purposes
datetime_name=`date '+%Y-%m-%d_%H:%M:%S'`

set -x

SLURM_JOB_ID=${SLURM_JOB_ID:=$datetime_name}

# path to project root
base_dir=${base_dir:=..}

# path to nematus (relative to root dir project)
nematus=${nematus:=nematus}

# transform params
tag_unit=${tag_unit:=char}
context_unit=${context_unit:=char}
context_size=${context_size:=20}
char_n_gram=${char_n_gram:=1}

# training params
patience=${patience:=2}
enc_depth=${enc_depth:=2}
dec_depth=${dec_depth:=2}
embedding_size=${embedding_size:=300}
state_size=${state_size:=100}

set +x
currentdir=`pwd`
cd $base_dir

echo Transforming
for partition in training dev test ; do
	input_file=${original_dataset}/${partition}.txt
	echo Transforming ${input_file}
	transform_folder_path=$( python -m data.transform_ud \
	--input $input_file \
	--tag_unit $tag_unit \
	--context_unit $context_unit \
	--context_size $context_size \
	--char_n_gram $char_n_gram \
	2> /dev/null | sed -n 1p )
	
	[ -z "$transform_folder_path" ] && echo "No transform folder generated. Exiting." && exit
done

set -x
transform_folder_name=$( basename $transform_folder_path )
model_name=m${enc_depth}_${dec_depth}_${embedding_size}_${state_size}
model_dir=models/${transform_folder_name}/$model_name
set +x

echo Copying data
mkdir -p models/
mkdir -p $model_dir
mkdir -p $model_dir/data

cp $transform_folder_path/* $model_dir/data/

# build dictionary
echo Building dictionaries
python ${nematus}/data/build_dictionary.py ${model_dir}/data/training_source ${model_dir}/data/training_target

echo Training
python ${nematus}/nematus/nmt.py \
--model ${model_dir}/model.npz \
--source_dataset ${model_dir}/data/training_source \
--target_dataset ${model_dir}/data/training_target \
--valid_source_dataset ${model_dir}/data/dev_source \
--valid_target_dataset ${model_dir}/data/dev_target \
--patience ${patience} \
--validFreq 1000 \
--saveFreq 0 \
--maxlen 50 \
--dispFreq 500 \
--batch_size=60 \
--enc_depth $enc_depth \
--dec_depth $dec_depth \
--embedding_size ${embedding_size} \
--state_size ${state_size} \
--dictionaries ${model_dir}/data/training_source.json ${model_dir}/data/training_target.json \
&> ${model_dir}/train-${SLURM_JOB_ID}.out

echo Translating dev set
python ${nematus}/nematus/translate.py \
-m ${model_dir}/model.npz \
-i ${model_dir}/data/dev_source \
-o ${model_dir}/data/dev_hypothesis \
-k 12 -n -p 1 \
&> ${model_dir}/translate-${SLURM_JOB_ID}.out

echo Postprocessing dev predictions
python -m data.postprocess_nematus ${model_dir}/data/dev_hypothesis data/datasets/MorphoData-NewSplit/dev.txt > ${model_dir}/data/dev_prediction

echo Calculating score
python -m analysis.score_prediction ${model_dir}/data/dev_prediction > ${model_dir}/data/dev_score

cd $currentdir