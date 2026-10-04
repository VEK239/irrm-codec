#!/bin/bash -l
#SBATCH --job-name=irrm-ylq-motif-eval
#SBATCH --partition=medium
#SBATCH --constraint=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
PREFLIGHT_DIR="${1:?Official-motif pooled preflight directory is required}"
ROOT="${RTP_CODEC_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
OUTPUT="$ROOT/experiments/trb/vdjdb-ylq-official-motif-pooled/evaluation-${SLURM_JOB_ID}"
TOKENIZER="$ROOT/experiments/trb/tokenizers/anchored-seed42/data_anchor/anchored_tokenizer.json"
PYTHON="${RTP_CODEC_PYTHON:-python}"
test -f "$PREFLIGHT_DIR/PREFLIGHT.json"
cd "$ROOT"
"$PYTHON" -m rtp_codec.benchmarks.downstream.evaluate_vdjdb_ylq_motif_pooled \
  --preflight-dir "$PREFLIGHT_DIR" --tokenizer "$TOKENIZER" --output-dir "$OUTPUT" \
  --model "r:$ROOT/experiments/trb/data-anchor-r-seed42/best.pt:$ROOT/experiments/trb/data-anchor-r-seed42/run_config.json" \
  --model "t:$ROOT/experiments/trb/data-anchor-t-seed42/best.pt:$ROOT/experiments/trb/data-anchor-t-seed42/run_config.json" \
  --model "p:$ROOT/experiments/trb/data-anchor-p-seed42/best.pt:$ROOT/experiments/trb/data-anchor-p-seed42/run_config.json" \
  --model "rt:$ROOT/experiments/trb/data-anchor-rt-seed42/best.pt:$ROOT/experiments/trb/data-anchor-rt-seed42/run_config.json" \
  --model "rp:$ROOT/experiments/trb/data-anchor-rp-seed42/best.pt:$ROOT/experiments/trb/data-anchor-rp-seed42/run_config.json" \
  --model "tp:$ROOT/experiments/trb/data-anchor-tp-seed42/best.pt:$ROOT/experiments/trb/data-anchor-tp-seed42/run_config.json" \
  --model "rtp:$ROOT/experiments/trb/anchor-data-rtp-seed42/best.pt:$ROOT/experiments/trb/anchor-data-rtp-seed42/run_config.json" \
  --batch-size 128 --bootstrap 2000 --permutations 1000 --seed 42
