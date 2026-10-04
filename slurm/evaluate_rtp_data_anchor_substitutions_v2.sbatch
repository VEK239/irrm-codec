#!/bin/bash -l
#SBATCH --job-name=irrm-rtp-subst
#SBATCH --partition=medium
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/evlasova/irrm-codec/experiments/trb/slurm-%x-%j.out
#SBATCH --error=/home/evlasova/irrm-codec/experiments/trb/slurm-%x-%j.err

set -euo pipefail

ROOT_DIR="/home/evlasova/irrm-codec"
DATA_DIR="$ROOT_DIR/data/benchmark/trb"
TOKENIZER="$ROOT_DIR/experiments/trb/tokenizers/anchored-seed42/data_anchor/anchored_tokenizer.json"
CHECKPOINT="$ROOT_DIR/experiments/trb/anchor-data-rtp-seed42/best.pt"
OUTPUT_DIR="$ROOT_DIR/experiments/trb/rtp-data-anchor-substitutions-$SLURM_JOB_ID"
PYTHON_BIN="/home/evlasova/.conda/envs/irrm-codec/bin/python"

test -x "$PYTHON_BIN"
test -f "$DATA_DIR/READY.json"
test -f "$TOKENIZER"
test -f "$CHECKPOINT"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" benchmark/evaluate_reconstruction_substitutions.py \
  --data-dir "$DATA_DIR" \
  --tokenizer "$TOKENIZER" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --batch-size 64 \
  --num-workers 8 \
  --device cuda
