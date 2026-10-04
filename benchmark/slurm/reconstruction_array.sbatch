#!/usr/bin/env bash
# Issue 2: train the IRRM-CODEC decoder for every representation and seed.
#
# 5 representations x 3 seeds = 15 tasks. The array is throttled to 2 concurrent
# tasks because the cluster allows at most 2 GPUs per user by default.
#
# The log directory must exist before submitting: Slurm opens the --output file before
# the script body runs, so a missing directory kills the job immediately.
#
# Submit from the repository root:
#   mkdir -p /mnt/tank/scratch/mpodsytnik/irrm-codec/logs
#   sbatch benchmark/slurm/reconstruction_array.sbatch
#
# Override anything without editing the file:
#   sbatch --export=ALL,BOTTLENECK_DIM=32,TRAIN_SUBSET=10k \
#     benchmark/slurm/reconstruction_array.sbatch

#SBATCH --job-name=irrm-recon
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
# A 60-epoch run takes ~40 min on a GTX 1080 Ti (39 s/epoch at train=79730), so this
# leaves ample headroom while keeping the job small enough to schedule quickly.
#SBATCH --time=02:00:00
#SBATCH --array=0-14%2
#SBATCH --output=/mnt/tank/scratch/mpodsytnik/irrm-codec/logs/%x_%A_%a.out
#SBATCH --error=/mnt/tank/scratch/mpodsytnik/irrm-codec/logs/%x_%A_%a.err

set -euo pipefail

SCRATCH="${SCRATCH:-/mnt/tank/scratch/mpodsytnik/irrm-codec}"
REPO="${REPO:-/nfs/home/mpodsytnik/irrm-codec}"
CONDA_ROOT="${CONDA_ROOT:-/mnt/tank/scratch/mpodsytnik/miniconda3}"
CONDA_ENV="${CONDA_ENV:-irrm-codec}"

DATASET_DIR="${DATASET_DIR:-$SCRATCH/data/benchmark/trb}"
BOTTLENECK_DIR="${BOTTLENECK_DIR:-$SCRATCH/data/benchmark/trb/bottleneck}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH/artifacts/benchmark/reconstruction}"

BOTTLENECK_DIM="${BOTTLENECK_DIM:-64}"
TRAIN_SUBSET="${TRAIN_SUBSET:-all}"
EPOCHS="${EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-256}"
PATIENCE="${PATIENCE:-8}"

# Task id maps to (representation, seed): ids 0-2 are the first representation, and so on.
REPRESENTATIONS=(tcremp onehot tcr_bert sceptr esm2_8m)
SEEDS=(42 43 44)
REPRESENTATION="${REPRESENTATIONS[$((SLURM_ARRAY_TASK_ID / ${#SEEDS[@]}))]}"
SEED="${SEEDS[$((SLURM_ARRAY_TASK_ID % ${#SEEDS[@]}))]}"

mkdir -p "$OUTPUT_DIR"

# conda init is deliberately not used: it would edit the shared account's ~/.bashrc.
source "$CONDA_ROOT/bin/activate"
conda activate "$CONDA_ENV"

echo "job=${SLURM_JOB_ID:-local} task=${SLURM_ARRAY_TASK_ID:-0} node=${SLURMD_NODENAME:-?}"
echo "representation=$REPRESENTATION seed=$SEED dim=$BOTTLENECK_DIM subset=$TRAIN_SUBSET"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cd "$REPO"
# Slurm redirects stdout to a file, where Python block-buffers by default and the log
# appears only in bursts. Unbuffered output keeps the .out file usable for monitoring.
export PYTHONUNBUFFERED=1
# Called directly rather than through srun: the batch task already holds its own
# allocation, and an inner step has to request the GPU again, which fails outright when
# the per-user GPU quota is busy elsewhere.
python -m benchmark.train_decoder \
  --dataset-dir "$DATASET_DIR" \
  --bottleneck-dir "$BOTTLENECK_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --representation "$REPRESENTATION" \
  --bottleneck-dim "$BOTTLENECK_DIM" \
  --train-subset "$TRAIN_SUBSET" \
  --seed "$SEED" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --patience "$PATIENCE" \
  --threads "${SLURM_CPUS_PER_TASK:-4}"
