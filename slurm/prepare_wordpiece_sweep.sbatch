#!/bin/bash -l
#SBATCH --job-name=irrm-wp-sweep-preflight
#SBATCH --partition=short
#SBATCH --constraint=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/evlasova/irrm-codec/experiments/trb/tokenizers/wordpiece-sweep-seed42/slurm-%x-%j.out
#SBATCH --error=/home/evlasova/irrm-codec/experiments/trb/tokenizers/wordpiece-sweep-seed42/slurm-%x-%j.err

set -euo pipefail
ROOT_DIR="/home/evlasova/irrm-codec"
DATA_DIR="$ROOT_DIR/data/benchmark/trb"
TOKENIZER_ROOT="$ROOT_DIR/experiments/trb/tokenizers/wordpiece-sweep-seed42"
PYTHON_BIN="/home/evlasova/.conda/envs/irrm-codec/bin/python"

test -f "$DATA_DIR/READY.json"
test ! -e "$TOKENIZER_ROOT/sweep_manifest.json"
cd "$ROOT_DIR"
"$PYTHON_BIN" -m scripts.prepare_wordpiece_sweep \
  --data-dir "$DATA_DIR" \
  --output-root "$TOKENIZER_ROOT" \
  --vocab-sizes 44 64 128 256 512 1024 2048 4096 \
  --min-frequency 2 \
  --seed 42
"$PYTHON_BIN" -m scripts.preflight_wordpiece_sweep \
  --data-dir "$DATA_DIR" \
  --tokenizer-root "$TOKENIZER_ROOT" \
  --baseline-run-config "$ROOT_DIR/experiments/trb/rq5-char-rtp-d128-seed42/run_config.json" \
  --output-path "$TOKENIZER_ROOT/preflight.json"
