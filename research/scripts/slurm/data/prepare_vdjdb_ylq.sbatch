#!/bin/bash -l
#SBATCH --job-name=irrm-ylq-preflight
#SBATCH --partition=short
#SBATCH --constraint=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
ROOT="${RTP_CODEC_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
OUTPUT="$ROOT/experiments/trb/vdjdb-ylq-data-anchor-validation/preflight-${SLURM_JOB_ID}"
PYTHON="${RTP_CODEC_PYTHON:-python}"
mkdir -p "$ROOT/experiments/trb/vdjdb-ylq-data-anchor-validation"
cd "$ROOT"
"$PYTHON" -m rtp_codec.benchmarks.datasets.prepare_vdjdb_ylq \
  --vdjdb-tsv /projects/immunestatus/vdjdb/vdjdb-2025-07-30/vdjdb.txt \
  --benchmark-dir "$ROOT/data/benchmark/trb" --output-dir "$OUTPUT" \
  --min-positive-cdr3 50 --min-positive-donors 2 --seed 42
