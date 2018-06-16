#!/bin/bash
. slurm_init.sh

mkdir -p logs

model_dir="models/test_model/sample_run" sbatch --output="logs/translate-%j.out" translate.sh