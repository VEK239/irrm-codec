#!/usr/bin/env bash
#SBATCH --job-name=irrm-multitask
#SBATCH --partition=medium
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"
export DATA_DIR="${DATA_DIR:-data/benchmark/trb}"
export OUTPUT_DIR="${OUTPUT_DIR:-artifacts/multitask/char}"
export TRAIN_SUBSET="${TRAIN_SUBSET:-all}"
export TOKENIZER_TYPE="${TOKENIZER_TYPE:-char}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export EPOCHS="${EPOCHS:-40}"
export SEED="${SEED:-42}"

bash scripts/train_multitask.sh --num-workers "${SLURM_CPUS_PER_TASK:-8}"
