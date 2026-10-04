#!/bin/bash -l
#SBATCH --job-name=irrm-data-factorial-pre
#SBATCH --partition=short
#SBATCH --constraint=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
ROOT_DIR="${RTP_CODEC_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
DATA_DIR="$ROOT_DIR/data/benchmark/trb"
TOKENIZER="$ROOT_DIR/experiments/trb/tokenizers/anchored-seed42/data_anchor/anchored_tokenizer.json"
REFERENCE="$ROOT_DIR/experiments/trb/anchor-data-rtp-seed42/run_config.json"
OUTPUT_ROOT="$ROOT_DIR/experiments/trb/data-anchor-full-factorial-gate"
PYTHON_BIN="${RTP_CODEC_PYTHON:-$(command -v python)}"

test -x "$PYTHON_BIN"
test -f "$DATA_DIR/READY.json"
test -f "$TOKENIZER"
test -f "$REFERENCE"
test ! -e "$OUTPUT_ROOT/PREFLIGHT.json"
mkdir -p "$OUTPUT_ROOT"
cd "$ROOT_DIR"
"$PYTHON_BIN" -m rtp_codec.experiments.preflight.preflight_data_anchor_objectives \
  --data-dir "$DATA_DIR" \
  --tokenizer "$TOKENIZER" \
  --reference-run-config "$REFERENCE" \
  --output-root "$OUTPUT_ROOT"
