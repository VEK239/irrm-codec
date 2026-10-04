#!/bin/bash -l
#SBATCH --job-name=vdjdb-umap
#SBATCH --partition=short
#SBATCH --constraint=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=umap-%j.out
#SBATCH --error=umap-%j.err

set -euo pipefail
repo=/home/evlasova/irrm-codec
out=/home/evlasova/irrm-codec/experiments/trb/vdjdb-uniform100-v1/umap-${SLURM_JOB_ID}
cd "$repo"
${RTP_CODEC_PYTHON:-python} -m rtp_codec.benchmarks.downstream.create_vdjdb_umap_candidates \
  --cohort /home/evlasova/irrm-codec/experiments/trb/vdjdb-redcea-clonotypes/preflight-1426719/cohort/cohort.tsv \
  --native-evaluation /home/evlasova/irrm-codec/experiments/trb/vdjdb-redcea-clonotypes/evaluation-1426728 \
  --output-dir "$out"
