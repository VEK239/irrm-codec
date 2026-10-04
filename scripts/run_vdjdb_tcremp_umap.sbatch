#!/bin/bash -l
#SBATCH --job-name=vdjdb-tcremp-umap
#SBATCH --partition=medium
#SBATCH --constraint=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/evlasova/irrm-codec/experiments/trb/vdjdb-uniform100-v1/tcremp-umap-%j.out
#SBATCH --error=/home/evlasova/irrm-codec/experiments/trb/vdjdb-uniform100-v1/tcremp-umap-%j.err

set -euo pipefail
repo=/home/evlasova/irrm-codec
runtime=/home/evlasova/irrm-codec/experiments/trb/vdjdb-uniform100-v1/umap-runtime-v1
out=/home/evlasova/irrm-codec/experiments/trb/vdjdb-uniform100-v1/tcremp-umap-${SLURM_JOB_ID}
cd "$repo"
"$runtime/bin/python" benchmark/create_vdjdb_tcremp_umap.py \
  --cohort /home/evlasova/irrm-codec/experiments/trb/vdjdb-redcea-clonotypes/preflight-1426719/cohort/cohort.tsv \
  --tcremp /home/evlasova/irrm-codec/experiments/trb/vdjdb-uniform100-v1/tcremp-1429926/tcremp.npy \
  --output-dir "$out"
