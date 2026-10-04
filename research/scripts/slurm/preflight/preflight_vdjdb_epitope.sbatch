#!/bin/bash -l
#SBATCH --job-name=irrm-vdjdb-preflight
#SBATCH --partition=short
#SBATCH --constraint=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
ROOT_DIR="${RTP_CODEC_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
OUTPUT_ROOT="$ROOT_DIR/experiments/trb/vdjdb-data-anchor-validation"
OUTPUT_DIR="$OUTPUT_ROOT/cohort-preflight-${SLURM_JOB_ID}"
VDJDB_SOURCE="/projects/immunestatus/vdjdb/vdjdb-2025-07-30/vdjdb.txt"
BENCHMARK_DIR="$ROOT_DIR/data/benchmark/trb"
EXPERIMENT_DIR="$ROOT_DIR/experiments/trb/anchor-data-rtp-seed42"
TOKENIZER="$ROOT_DIR/experiments/trb/tokenizers/anchored-seed42/data_anchor/anchored_tokenizer.json"
PYTHON_BIN="${RTP_CODEC_PYTHON:-$(command -v python)}"

test -x "$PYTHON_BIN"
test -f "$VDJDB_SOURCE"
test -f "$BENCHMARK_DIR/READY.json"
test -f "$EXPERIMENT_DIR/best.pt"
test -f "$EXPERIMENT_DIR/run_config.json"
test -f "$TOKENIZER"
mkdir -p "$OUTPUT_ROOT"
cd "$ROOT_DIR"
"$PYTHON_BIN" -m rtp_codec.benchmarks.datasets.prepare_vdjdb_epitope_cohort \
  --vdjdb-tsv "$VDJDB_SOURCE" \
  --benchmark-dir "$BENCHMARK_DIR" \
  --checkpoint "$EXPERIMENT_DIR/best.pt" \
  --run-config "$EXPERIMENT_DIR/run_config.json" \
  --tokenizer "$TOKENIZER" \
  --output-dir "$OUTPUT_DIR" \
  --min-support 50 --min-donors 2 \
  --sensitivity-thresholds 25 50 100 \
  --min-vdjdb-score 1 --seed 42
