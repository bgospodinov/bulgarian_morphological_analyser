#!/bin/bash
source slurm_init.sh
cd ..

# path to original dataset (relative to root dir of project)
original_dataset=data/datasets/MorphoData-NewSplit

for partition in training dev test ; do
	input_file=${original_dataset}/${partition}.txt
	echo Transforming ${input_file}
	transform_folder_path=$( python -m data.transform_ud --input $input_file | sed -n 1p )
done

transform_folder_name=$( basename $transform_folder_path )

# path to nematus (relative to root dir project)
nematus=nematus

# folder name for the model
model_name=test-model
model_dir=models/$model_name

mkdir -p models/
mkdir -p $model_dir
mkdir -p $model_dir/data

cp $transform_folder_path/* $model_dir/data/

# build dictionary
echo Building dictionaries
python ${nematus}/data/build_dictionary.py ${model_dir}/data/training_source ${model_dir}/data/training_target

# training
echo Training

python ${nematus}/nematus/nmt.py \
--model ${model_dir}/model.npz \
--source_dataset ${model_dir}/data/training_source \
--target_dataset ${model_dir}/data/training_target \
--valid_source_dataset ${model_dir}/data/dev_source \
--valid_target_dataset ${model_dir}/data/dev_target \
--patience 2 \
--validFreq 5000 \
--saveFreq 0 \
--dictionaries ${model_dir}/data/training_source.json ${model_dir}/data/training_target.json > ${model_dir}/`date '+%Y-%m-%d-%H:%M:%S'`.out

cd scripts/