#!/bin/bash
. scripts/slurm_init.sh

mkdir -p logs

model_dir="models/test_model/sample_run" sbatch -v --output="logs/translate-%u-%j.out" --error="logs/translate-%u-%j.err" scripts/translate.sh