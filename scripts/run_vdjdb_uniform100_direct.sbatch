#!/bin/bash -l
#SBATCH --job-name=vdjdb-uniform100
#SBATCH --partition=medium
#SBATCH --constraint=compute
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/evlasova/irrm-codec/experiments/trb/vdjdb-uniform100-v1/slurm-%j.out
#SBATCH --error=/home/evlasova/irrm-codec/experiments/trb/vdjdb-uniform100-v1/slurm-%j.err

set -euo pipefail

repo=/home/evlasova/irrm-codec
out=/home/evlasova/irrm-codec/experiments/trb/vdjdb-uniform100-v1/evaluation-${SLURM_JOB_ID}

cd "$repo"
/home/evlasova/.conda/envs/irrm-codec/bin/python benchmark/evaluate_vdjdb_uniform100.py \
  --cohort /home/evlasova/irrm-codec/experiments/trb/vdjdb-redcea-clonotypes/preflight-1426719/cohort/cohort.tsv \
  --native-evaluation /home/evlasova/irrm-codec/experiments/trb/vdjdb-redcea-clonotypes/evaluation-1426728 \
  --external-representations /home/evlasova/irrm-codec/experiments/trb/external-global-author-v1/vdjdb/representations \
  --output-dir "$out"
