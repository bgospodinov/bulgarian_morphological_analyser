#!/bin/bash
. slurm_init.sh

mkdir -p logs

model_dir="models/test-model" sbatch --output="logs/translate-%j.out" translate.sh