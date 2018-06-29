#!/bin/bash
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --mem=8000  # memory in Mb

# Author: Bogomil Gospodinov
# path to original dataset (relative to root dir of project)
original_dataset=data/datasets/MorphoData-NewSplit

# if slurm job id is unavailable use datetime for logging purposes
datetime_name=`date +%s`

set -x

SLURMD_NODENAME=${SLURMD_NODENAME}
SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}
SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION}
SLURM_ENABLED=${SLURM_JOB_ID:+1}
SLURM_JOB_ID=${SLURM_JOB_ID:=$datetime_name}
SLURM_ORIGINAL_JOB_ID=${SLURM_ORIGINAL_JOB_ID}
skip_resume_training=${skip_resume_training:+1}
seed=${seed:=0}

# path to nematus (relative to root dir project)
nematus=${nematus:=nematus}

set +x

if [[ -z "$SLURM_ORIGINAL_JOB_ID" ]]; then
	set -x

	# transform params
	tag_unit=${tag_unit:=char}
	context_unit=${context_unit:=char}
	context_size=${context_size:=20}
	char_n_gram=${char_n_gram:=1}
	transform_mode=${transform_mode:=word_and_context}
	output_dir=${TMPDIR:-data/input/}

	# training params
	patience=${patience:=2}
	optimizer=${optimizer:=adadelta}
	learning_rate=${learning_rate:=1.0}
	enc_depth=${enc_depth:=3}
	dec_depth=${dec_depth:=1}
	embedding_size=${embedding_size:=300}
	state_size=${state_size:=300}
	dropout_embedding=${dropout_embedding:=0.2}
	dropout_hidden=${dropout_hidden:=0.3}
	dropout_source=${dropout_source:=0.0}
	dropout_target=${dropout_target:=0.0}
	output_hidden_activation=${output_hidden_activation:=tanh}
	decay_c=${decay_c:=0.0}

	set +x

	if [[ -n "$SLURM_ENABLED" ]]; then
		/usr/bin/time -f %e cp -urv ${original_dataset} ${TMPDIR}
		set -x
		original_dataset=${TMPDIR}/${original_dataset##*/}
		set +x
	fi

	echo Transforming
	for partition in training dev test ; do
		input_file=${original_dataset}/${partition}.txt
		echo Transforming ${input_file}
		set -x

		transform_folder_path=$( /usr/bin/time -f %e python -m data.transform_ud \
		--input $input_file \
		--output $output_dir \
		--mode $transform_mode \
		--tag_unit $tag_unit \
		--context_unit $context_unit \
		--context_size $context_size \
		--char_n_gram_mode $char_n_gram \
		--transform_appendix $SLURM_JOB_ID \
		| sed -n 1p )

		set +x
		
		[ -z "$transform_folder_path" ] && echo "No transform folder found or generated. Exiting." && exit
	done

	if [[ -n "$SLURM_ENABLED" ]]; then
		term_handler()
		{
			echo "function term_handler called. Exiting..."
			# do whatever cleanup you want here
			rm -rfv $transform_folder_path
			echo $transform_folder_path deleted
		}
		# associate the function "term_handler" with the TERM or EXIT signals
		trap 'term_handler' TERM EXIT
	fi

	set -x
	transform_folder_name=$( basename $transform_folder_path )
	model_name=m${enc_depth}_${dec_depth}_${embedding_size}_${state_size}_${output_hidden_activation}_${decay_c}_${dropout_embedding}_${dropout_hidden}_${dropout_source}_${dropout_target}_${optimizer}_${learning_rate}
	model_dir=models/${transform_folder_name%.*}/$model_name
	set +x

	echo Copying data
	mkdir -p models/
	mkdir -p $model_dir
	mkdir -p $model_dir/${SLURM_JOB_ID}
	mkdir -p $model_dir/data

	/usr/bin/time -f %e cp -n $transform_folder_path/* $model_dir/data/

	# build dictionaries only if they dont exist
	if [[ ! -f $model_dir/data/training_source.json && ! -f $model_dir/data/training_target.json ]]; then
		echo Building dictionaries
		/usr/bin/time -f %e python ${nematus}/data/build_dictionary.py ${model_dir}/data/training_source ${model_dir}/data/training_target
	else
		echo Dictionaries found and reused
	fi

	echo Training
	/usr/bin/time -f %e python ${nematus}/nematus/nmt.py \
	--model ${model_dir}/${SLURM_JOB_ID}/model.npz \
	--source_dataset ${model_dir}/data/training_source \
	--target_dataset ${model_dir}/data/training_target \
	--valid_source_dataset ${model_dir}/data/dev_source \
	--valid_target_dataset ${model_dir}/data/dev_target \
	--keep_train_set_in_memory \
	--patience ${patience} \
	--validFreq 3000 \
	--validBurnIn 10000 \
	--saveFreq 0 \
	--maxlen 50 \
	--dispFreq 100 \
	--sampleFreq 100 \
	--batch_size 60 \
	--use_dropout \
	--dropout_embedding ${dropout_embedding} \
	--dropout_hidden ${dropout_hidden} \
	--dropout_source ${dropout_source} \
	--dropout_target ${dropout_target} \
	--optimizer ${optimizer} \
	--learning_rate ${learning_rate} \
	--decay_c ${decay_c} \
	--enc_depth $enc_depth \
	--dec_depth $dec_depth \
	--embedding_size ${embedding_size} \
	--state_size ${state_size} \
	--output_hidden_activation ${output_hidden_activation} \
	--random_seed "${seed}" \
	--dictionaries ${model_dir}/data/training_source.json ${model_dir}/data/training_target.json

	if [ $? -ne 0 ]; then
		exit -1
	fi

else
	echo Reloading model for job $SLURM_ORIGINAL_JOB_ID
	set -x
	SLURM_JOB_ID=$SLURM_ORIGINAL_JOB_ID
	model_dir=$( find models/ -path "*/$SLURM_ORIGINAL_JOB_ID" | head -n 1 )
	model_dir=${model_dir%/*}
	set +x
	if [ -z "$model_dir" ]; then
		echo No models for $SLURM_ORIGINAL_JOB_ID found in models/.
		exit -1
	fi

	if [ -z "$skip_resume_training" ]; then
		echo Resuming training
		/usr/bin/time -f %e python ${nematus}/nematus/nmt.py \
		--model ${model_dir}/${SLURM_JOB_ID}/model.npz \
		--load_model_config \
		--reload latest_checkpoint
	else
		echo Skipping over training...
	fi
fi

echo Translating dev set
/usr/bin/time -f %e python ${nematus}/nematus/translate.py \
-m ${model_dir}/${SLURM_JOB_ID}/model.npz \
-i ${model_dir}/data/dev_source \
-o ${model_dir}/data/dev_hypothesis.${SLURM_JOB_ID} \
-k 12 -n -p 1 -v

echo Postprocessing dev predictions
/usr/bin/time -f %e python -m data.postprocess_nematus ${model_dir}/data/dev_hypothesis.${SLURM_JOB_ID} data/datasets/MorphoData-NewSplit/dev.txt > ${model_dir}/data/dev_prediction.${SLURM_JOB_ID}

echo Calculating score
/usr/bin/time -f %e python -m analysis.score_prediction ${model_dir}/data/dev_prediction.${SLURM_JOB_ID} > ${model_dir}/data/dev_score.${SLURM_JOB_ID}

echo Concatenating
cat ${model_dir}/data/dev_score.* > ${model_dir}/data/dev_scores

echo Averaging
/usr/bin/time -f %e python -m analysis.average ${model_dir}/data/dev_scores > ${model_dir}/data/dev_avg_score