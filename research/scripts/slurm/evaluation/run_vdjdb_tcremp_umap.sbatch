#!/bin/bash -l
#SBATCH --job-name=vdjdb-tcremp-umap
#SBATCH --partition=medium
#SBATCH --constraint=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=tcremp-umap-%j.out
#SBATCH --error=tcremp-umap-%j.err

set -euo pipefail
repo=/home/evlasova/irrm-codec
runtime=/home/evlasova/irrm-codec/experiments/trb/vdjdb-uniform100-v1/umap-runtime-v1
out=/home/evlasova/irrm-codec/experiments/trb/vdjdb-uniform100-v1/tcremp-umap-${SLURM_JOB_ID}
cd "$repo"
"$runtime/bin/python" -m rtp_codec.benchmarks.downstream.create_vdjdb_tcremp_umap \
  --cohort /home/evlasova/irrm-codec/experiments/trb/vdjdb-redcea-clonotypes/preflight-1426719/cohort/cohort.tsv \
  --tcremp /home/evlasova/irrm-codec/experiments/trb/vdjdb-uniform100-v1/tcremp-1429926/tcremp.npy \
  --output-dir "$out"
