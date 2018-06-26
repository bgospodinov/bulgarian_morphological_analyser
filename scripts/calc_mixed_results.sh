#!/bin/bash

initial_slurm_id=130121
model_dir=models/MorphoData-NewSplit_20_char_1/m3_1_300_300_adadelta_1.0/data
number_of_experiments=8
runs=3

for (( i=0; i<$number_of_experiments; i++ )) ; do 
	python -m analysis.average <( cat $(for (( j=0; j<$runs; j++ )) ; do echo ${model_dir}/dev_score.$(( $initial_slurm_id + $i + $j * $number_of_experiments )) ; done) ) ; 
done
