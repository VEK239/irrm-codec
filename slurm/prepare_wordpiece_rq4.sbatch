#!/bin/bash -l
#SBATCH --job-name=irrm-rq4-wp-preflight
#SBATCH --partition=short
#SBATCH --constraint=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/evlasova/irrm-codec/experiments/trb/tokenizers/wordpiece-seed42/slurm-%x-%j.out
#SBATCH --error=/home/evlasova/irrm-codec/experiments/trb/tokenizers/wordpiece-seed42/slurm-%x-%j.err

set -euo pipefail
ROOT_DIR="/home/evlasova/irrm-codec"
DATA_DIR="$ROOT_DIR/data/benchmark/trb"
TOKENIZER_DIR="$ROOT_DIR/experiments/trb/tokenizers/wordpiece-seed42"
PYTHON_BIN="/home/evlasova/.conda/envs/irrm-codec/bin/python"

test -f "$DATA_DIR/READY.json"
test ! -e "$TOKENIZER_DIR/tokenizer.json"
cd "$ROOT_DIR"
"$PYTHON_BIN" -m scripts.prepare_wordpiece_rq4 \
  --data-dir "$DATA_DIR" \
  --output-dir "$TOKENIZER_DIR" \
  --vocab-size 256 \
  --min-frequency 2 \
  --seed 42
"$PYTHON_BIN" -m scripts.preflight_wordpiece_rq4 \
  --data-dir "$DATA_DIR" \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --output-path "$TOKENIZER_DIR/preflight.json"
