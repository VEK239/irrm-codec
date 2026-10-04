#!/usr/bin/env bash
#SBATCH --job-name=rtp-codec
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
ARM="${1:-rtp}"
case "$ARM" in r|t|p|rt|rp|tp|rtp) ;; *) echo "Unknown objective condition: $ARM" >&2; exit 2 ;; esac
ROOT="${RTP_CODEC_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON="${RTP_CODEC_PYTHON:-python}"
: "${DATA_DIR:?Set DATA_DIR to the locked prepared benchmark}"
: "${TOKENIZER_PATH:?Set TOKENIZER_PATH to the DATA-ANCHOR bundle}"
cd "$ROOT"
"$PYTHON" -m rtp_codec train --config "research/configs/paper/$ARM.json" \
  --data-dir "$DATA_DIR" --tokenizer-path "$TOKENIZER_PATH" \
  --output-dir "${OUTPUT_DIR:-artifacts/paper/$ARM-seed42}" \
  --device cuda --num-workers "${SLURM_CPUS_PER_TASK:-8}"
