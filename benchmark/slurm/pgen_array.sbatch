#!/usr/bin/env bash
# Issue 3: every Pgen arm across both targets, three training sizes and three seeds.
#
# 5 neural arms x 2 targets x 3 subsets x 3 seeds = 90 runs, plus tfidf_ridge which is
# deterministic and needs one run per target and subset = 6. Total 96 tasks, throttled to
# 2 concurrent because the cluster allows at most 2 GPUs per user.
#
# The log directory must exist before submitting: Slurm opens the --output file before the
# script body runs, so a missing directory kills the job immediately.
#
#   mkdir -p /mnt/tank/scratch/mpodsytnik/irrm-codec/logs
#   sbatch benchmark/slurm/pgen_array.sbatch
#
# tfidf_ridge does not need a GPU, but it finishes in seconds, so keeping one array is
# simpler than splitting it onto the CPU partition.

#SBATCH --job-name=irrm-pgen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --array=0-95%2
#SBATCH --output=/mnt/tank/scratch/mpodsytnik/irrm-codec/logs/%x_%A_%a.out
#SBATCH --error=/mnt/tank/scratch/mpodsytnik/irrm-codec/logs/%x_%A_%a.err

set -euo pipefail

SCRATCH="${SCRATCH:-/mnt/tank/scratch/mpodsytnik/irrm-codec}"
REPO="${REPO:-/nfs/home/mpodsytnik/irrm-codec}"
CONDA_ROOT="${CONDA_ROOT:-/mnt/tank/scratch/mpodsytnik/miniconda3}"
CONDA_ENV="${CONDA_ENV:-irrm-codec}"

DATASET_DIR="${DATASET_DIR:-$SCRATCH/data/benchmark/trb}"
REPRESENTATIONS_DIR="${REPRESENTATIONS_DIR:-$SCRATCH/data/benchmark/trb/representations}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-$SCRATCH/artifacts/benchmark/forward_encoder/best.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH/artifacts/benchmark/pgen}"

EPOCHS="${EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-256}"
# Left to the trainer's own default so the cosine schedule is not cut short.
PATIENCE="${PATIENCE:-12}"

# Build the task list in a fixed order so a task id always maps to the same run.
CONFIGS=()
for target in log10_pgen log10_pgen_1mm; do
  for subset in 1k 10k all; do
    for arm in irrm_scratch irrm_pretrained sceptr_mlp tcr_bert_mlp esm2_8m_mlp; do
      for seed in 42 43 44; do
        CONFIGS+=("$arm $target $subset $seed")
      done
    done
    CONFIGS+=("tfidf_ridge $target $subset 42")
  done
done

if [ "${SLURM_ARRAY_TASK_ID:-0}" -ge "${#CONFIGS[@]}" ]; then
  echo "task ${SLURM_ARRAY_TASK_ID} is past the end of the ${#CONFIGS[@]}-run list" >&2
  exit 1
fi
read -r ARM TARGET SUBSET SEED <<< "${CONFIGS[${SLURM_ARRAY_TASK_ID:-0}]}"

mkdir -p "$OUTPUT_DIR"

# conda init is deliberately not used: it would edit the shared account's ~/.bashrc.
source "$CONDA_ROOT/bin/activate"
conda activate "$CONDA_ENV"

echo "job=${SLURM_JOB_ID:-local} task=${SLURM_ARRAY_TASK_ID:-0} node=${SLURMD_NODENAME:-?}"
echo "arm=$ARM target=$TARGET subset=$SUBSET seed=$SEED (${#CONFIGS[@]} runs total)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cd "$REPO"
# Slurm redirects stdout to a file, where Python block-buffers by default.
export PYTHONUNBUFFERED=1
# Called directly rather than through srun: the batch task already holds its own
# allocation, and an inner step has to request the GPU again, which fails outright when
# the per-user GPU quota is busy elsewhere.
python -m benchmark.train_pgen_arm \
  --arm "$ARM" \
  --target "$TARGET" \
  --train-subset "$SUBSET" \
  --seed "$SEED" \
  --dataset-dir "$DATASET_DIR" \
  --representations-dir "$REPRESENTATIONS_DIR" \
  --pretrained-checkpoint "$PRETRAINED_CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --patience "$PATIENCE" \
  --threads "${SLURM_CPUS_PER_TASK:-4}"
