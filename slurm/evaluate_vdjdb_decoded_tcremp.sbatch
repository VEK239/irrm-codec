#!/bin/bash -l
#SBATCH --job-name=vdjdb-decoded-tcremp
#SBATCH --partition=medium
#SBATCH --constraint=compute
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=16:00:00

set -euo pipefail
ROOT=/home/evlasova/irrm-codec
OUT="$ROOT/experiments/trb/vdjdb_decoded_tcremp/${SLURM_JOB_ID}"
cd "$ROOT"
PYTHONPATH="$ROOT" /home/evlasova/.conda/envs/irrm-codec/bin/python benchmark/evaluate_vdjdb_decoded_tcremp.py \
  --cohort "$ROOT/experiments/trb/vdjdb-redcea-clonotypes/preflight-1426719/cohort/cohort.tsv" \
  --rtp "$ROOT/experiments/trb/vdjdb-redcea-clonotypes/evaluation-1426728/embeddings_rtp.npy" \
  --checkpoint "$ROOT/experiments/trb/anchor-data-rtp-seed42/best.pt" \
  --standardizer "$ROOT/experiments/trb/anchor-data-rtp-seed42/target_standardizer.npz" \
  --uniform-evaluation "$ROOT/experiments/trb/vdjdb-uniform100-v1/evaluation-1427813" \
  --existing-separation "$ROOT/experiments/trb/vdjdb-uniform100-v1/separation-1428106" \
  --native-tcremp "$ROOT/experiments/trb/vdjdb-uniform100-v1/tcremp-1429926/tcremp.npy" \
  --output-dir "$OUT" --batch-size 256
