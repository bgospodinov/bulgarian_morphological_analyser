#!/bin/bash

mkdir -p logs

for context_size in 10 15 20 25 ; do
	echo Launching\ context_size=$context_size
	context_size sbatch train.sh --output=logs/%j.out
done