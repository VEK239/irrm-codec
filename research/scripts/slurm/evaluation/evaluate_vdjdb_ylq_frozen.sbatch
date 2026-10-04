#!/bin/bash -l
#SBATCH --job-name=irrm-ylq-eval
#SBATCH --partition=medium
#SBATCH --constraint=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
PREFLIGHT_DIR="${1:?YLQ preflight directory is required}"
ROOT="${RTP_CODEC_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
OUTPUT="$ROOT/experiments/trb/vdjdb-ylq-data-anchor-validation/evaluation-${SLURM_JOB_ID}"
TOKENIZER="$ROOT/experiments/trb/tokenizers/anchored-seed42/data_anchor/anchored_tokenizer.json"
PYTHON="${RTP_CODEC_PYTHON:-python}"
test -f "$PREFLIGHT_DIR/PREFLIGHT.json"
cd "$ROOT"
"$PYTHON" -m rtp_codec.benchmarks.downstream.evaluate_vdjdb_ylq_frozen \
  --preflight-dir "$PREFLIGHT_DIR" --tokenizer "$TOKENIZER" --output-dir "$OUTPUT" \
  --model "r:$ROOT/experiments/trb/data-anchor-r-seed42/best.pt:$ROOT/experiments/trb/data-anchor-r-seed42/run_config.json" \
  --model "p:$ROOT/experiments/trb/data-anchor-p-seed42/best.pt:$ROOT/experiments/trb/data-anchor-p-seed42/run_config.json" \
  --model "rp:$ROOT/experiments/trb/data-anchor-rp-seed42/best.pt:$ROOT/experiments/trb/data-anchor-rp-seed42/run_config.json" \
  --model "rtp:$ROOT/experiments/trb/anchor-data-rtp-seed42/best.pt:$ROOT/experiments/trb/anchor-data-rtp-seed42/run_config.json" \
  --bootstrap 2000 --permutations 1000 --seed 42
