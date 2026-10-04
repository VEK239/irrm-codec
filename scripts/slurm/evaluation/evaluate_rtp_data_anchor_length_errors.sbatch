#!/bin/bash -l
#SBATCH --job-name=irrm-rtp-lenerr
#SBATCH --partition=medium
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
ROOT_DIR="${RTP_CODEC_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
DATA_DIR="$ROOT_DIR/data/benchmark/trb"
TOKENIZER="$ROOT_DIR/experiments/trb/tokenizers/anchored-seed42/data_anchor/anchored_tokenizer.json"
CHECKPOINT="$ROOT_DIR/experiments/trb/anchor-data-rtp-seed42/best.pt"
OUTPUT_DIR="$ROOT_DIR/experiments/trb/rtp-data-anchor-length-errors-$SLURM_JOB_ID"
PYTHON_BIN="${RTP_CODEC_PYTHON:-$(command -v python)}"

test -x "$PYTHON_BIN"
test -f "$DATA_DIR/READY.json"
test -f "$TOKENIZER"
test -f "$CHECKPOINT"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" -m rtp_codec.benchmarks.analysis.evaluate_rtp_reconstruction_by_length \
  --data-dir "$DATA_DIR" --tokenizer "$TOKENIZER" --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" --batch-size 64 --num-workers 8 --device cuda
