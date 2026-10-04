#!/bin/bash -l
#SBATCH --job-name=irrm-rq4-wp-preflight
#SBATCH --partition=short
#SBATCH --constraint=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
ROOT_DIR="${RTP_CODEC_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
DATA_DIR="$ROOT_DIR/data/benchmark/trb"
TOKENIZER_DIR="$ROOT_DIR/experiments/trb/tokenizers/wordpiece-seed42"
PYTHON_BIN="${RTP_CODEC_PYTHON:-$(command -v python)}"

test -f "$DATA_DIR/READY.json"
test ! -e "$TOKENIZER_DIR/tokenizer.json"
cd "$ROOT_DIR"
"$PYTHON_BIN" -m rtp_codec.experiments.tokenizers.prepare_wordpiece_rq4 \
  --data-dir "$DATA_DIR" \
  --output-dir "$TOKENIZER_DIR" \
  --vocab-size 256 \
  --min-frequency 2 \
  --seed 42
"$PYTHON_BIN" -m rtp_codec.experiments.preflight.preflight_wordpiece_rq4 \
  --data-dir "$DATA_DIR" \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --output-path "$TOKENIZER_DIR/preflight.json"
