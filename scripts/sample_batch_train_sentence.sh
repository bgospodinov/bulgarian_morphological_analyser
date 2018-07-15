#!/bin/bash
# Author: Bogomil Gospodinov
. scripts/slurm_init.sh

mkdir -p logs

runs=3

declare -a attrs=(500 1000 2000 5000 10000)

# 1 if the corresponding job should go to the LongJobs partition, 0 otherwise
declare -a lj_pred=(0 0 0 0 0)

for (( run=1; run<=$runs; run++ ))
do
	echo Run $run
	iter=0
	for attr1 in "${attrs[@]}" ; do
		#for attr2 in "${attrs[@]}" ; do
			echo Launching\ attribute=${attr1}

			word_unit=bpe bpe_operations=${attr1} output_hidden_activation=tanh enc_depth=1 dec_depth=1 dropout_embedding=0.1 dropout_hidden=0.2 embedding_size=400 state_size=300 patience=5 maxlen=70 valid_burn_in=0 valid_freq=300 valid_batch_size=25 transform_mode=sentence_to_sentence seed=$run sbatch \
				--output="logs/%x.%j.log" --error="logs/%x.%j.log" \
				--begin=now+0hour \
				--open-mode=append \
				--mail-type=END,FAIL --mail-user="$(whoami)@sms.ed.ac.uk" \
				--job-name="${run}_${attr1}_sent2sent" --partition=$( (( ${lj_pred[$iter]} == 1 )) && printf %s 'LongJobs' || printf %s 'Standard,Short' ) scripts/train.sh

			((iter++))
		#done
	done
done
