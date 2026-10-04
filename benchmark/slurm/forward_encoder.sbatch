#!/usr/bin/env bash
# Pretrain the sequence-to-TCRemP encoder that the irrm_pretrained Pgen arm starts from.
#
# Trained on the benchmark's own train split, so nothing the Pgen arms are evaluated on
# was seen during pretraining. Must finish before the Pgen arrays start.

#SBATCH --job-name=irrm-fwd
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/tank/scratch/mpodsytnik/irrm-codec/logs/%x_%j.out
#SBATCH --error=/mnt/tank/scratch/mpodsytnik/irrm-codec/logs/%x_%j.err

set -euo pipefail

SCRATCH="${SCRATCH:-/mnt/tank/scratch/mpodsytnik/irrm-codec}"
REPO="${REPO:-/nfs/home/mpodsytnik/irrm-codec}"
CONDA_ROOT="${CONDA_ROOT:-/mnt/tank/scratch/mpodsytnik/miniconda3}"
CONDA_ENV="${CONDA_ENV:-irrm-codec}"

DATASET_DIR="${DATASET_DIR:-$SCRATCH/data/benchmark/trb}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH/artifacts/benchmark/forward_encoder}"
EPOCHS="${EPOCHS:-40}"
PATIENCE="${PATIENCE:-12}"

source "$CONDA_ROOT/bin/activate"
conda activate "$CONDA_ENV"

echo "job=${SLURM_JOB_ID:-local} node=${SLURMD_NODENAME:-?}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cd "$REPO"
export PYTHONUNBUFFERED=1
python -m benchmark.train_forward_encoder \
  --dataset-dir "$DATASET_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --train-subset all \
  --epochs "$EPOCHS" \
  --patience "$PATIENCE" \
  --threads "${SLURM_CPUS_PER_TASK:-4}"
