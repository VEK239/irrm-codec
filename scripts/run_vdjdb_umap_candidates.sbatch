#!/bin/bash -l
#SBATCH --job-name=vdjdb-umap
#SBATCH --partition=short
#SBATCH --constraint=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/evlasova/irrm-codec/experiments/trb/vdjdb-uniform100-v1/umap-%j.out
#SBATCH --error=/home/evlasova/irrm-codec/experiments/trb/vdjdb-uniform100-v1/umap-%j.err

set -euo pipefail
repo=/home/evlasova/irrm-codec
out=/home/evlasova/irrm-codec/experiments/trb/vdjdb-uniform100-v1/umap-${SLURM_JOB_ID}
cd "$repo"
/home/evlasova/.conda/envs/irrm-codec/bin/python benchmark/create_vdjdb_umap_candidates.py \
  --cohort /home/evlasova/irrm-codec/experiments/trb/vdjdb-redcea-clonotypes/preflight-1426719/cohort/cohort.tsv \
  --native-evaluation /home/evlasova/irrm-codec/experiments/trb/vdjdb-redcea-clonotypes/evaluation-1426728 \
  --output-dir "$out"
