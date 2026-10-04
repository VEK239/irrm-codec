#!/bin/bash -l
#SBATCH --job-name=vdjdb-whole-umap
#SBATCH --partition=medium
#SBATCH --constraint=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=whole-umap-%j.out
#SBATCH --error=whole-umap-%j.err

set -euo pipefail
repo=/home/evlasova/irrm-codec
runtime=/home/evlasova/irrm-codec/experiments/trb/vdjdb-uniform100-v1/umap-runtime-v1
out=/home/evlasova/irrm-codec/experiments/trb/vdjdb-uniform100-v1/whole-umap-${SLURM_JOB_ID}
cd "$repo"
"$runtime/bin/python" -m rtp_codec.benchmarks.downstream.create_vdjdb_whole_umap \
  --cohort /home/evlasova/irrm-codec/experiments/trb/vdjdb-redcea-clonotypes/preflight-1426719/cohort/cohort.tsv \
  --native-evaluation /home/evlasova/irrm-codec/experiments/trb/vdjdb-redcea-clonotypes/evaluation-1426728 \
  --output-dir "$out"
