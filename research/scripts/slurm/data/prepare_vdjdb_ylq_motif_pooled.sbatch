#!/bin/bash -l
#SBATCH --job-name=irrm-ylq-motif-pre
#SBATCH --partition=short
#SBATCH --constraint=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
ROOT="${RTP_CODEC_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
SNAPSHOT="/projects/immunestatus/vdjdb/vdjdb-2025-07-30"
OUTPUT="$ROOT/experiments/trb/vdjdb-ylq-official-motif-pooled/preflight-${SLURM_JOB_ID}"
PYTHON="${RTP_CODEC_PYTHON:-python}"
mkdir -p "$ROOT/experiments/trb/vdjdb-ylq-official-motif-pooled"
cd "$ROOT"
"$PYTHON" -m rtp_codec.benchmarks.datasets.prepare_vdjdb_ylq_motif_pooled \
  --vdjdb-tsv "$SNAPSHOT/vdjdb.txt" \
  --cluster-members "$SNAPSHOT/cluster_members.txt" \
  --motif-pwms "$SNAPSHOT/motif_pwms.txt" \
  --benchmark-dir "$ROOT/data/benchmark/trb" \
  --output-dir "$OUTPUT" --min-score 1 --min-positives 40 --min-negatives 100 \
  --controls-per-positive 4 --seed 42
