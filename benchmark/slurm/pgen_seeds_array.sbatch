#!/usr/bin/env bash
# Issue 3, follow-up: extra seeds for the two IRRM arms on the full training split.
#
# The main array ran three seeds per arm. For the frozen-embedding arms that is plenty
# (their RMSE spread across seeds is about 0.005), but the IRRM arms train a full encoder
# and spread by 0.03-0.06 — larger than the scratch-vs-pretrained difference they are
# meant to measure. Seven more seeds take the standard error from 0.033 to 0.018 and make
# the pretraining question answerable.
#
# All three training sizes, so the low-data question gets the same statistical treatment
# as the full-data one: 2 arms x 2 targets x 3 subsets x 7 seeds. Results land in the same
# output directory, so collect_pgen picks them up with no changes.
#
#   mkdir -p /mnt/tank/scratch/mpodsytnik/irrm-codec/logs
#   sbatch benchmark/slurm/pgen_seeds_array.sbatch
#
# To repeat only some of them, override SUBSETS:
#   sbatch --export=ALL,SUBSETS="1k 10k" benchmark/slurm/pgen_seeds_array.sbatch

#SBATCH --job-name=irrm-pgen-seeds
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --array=0-55%2
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

# "all" is excluded by default because it was covered by an earlier submission of this
# same file; add it back through SUBSETS and widen --array to match if it is ever needed.
SUBSETS="${SUBSETS:-1k 10k}"
EPOCHS="${EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-256}"
# Left to the trainer's own default so the cosine schedule is not cut short.
PATIENCE="${PATIENCE:-12}"

# Seeds 42-44 already ran in the main array; these continue the sequence.
CONFIGS=()
for target in log10_pgen log10_pgen_1mm; do
  for subset in $SUBSETS; do
    for arm in irrm_scratch irrm_pretrained; do
      for seed in 45 46 47 48 49 50 51; do
        CONFIGS+=("$arm $target $subset $seed")
      done
    done
  done
done

# Exit cleanly rather than failing, so an array wider than the config list leaves no
# spurious FAILED tasks in the accounting record.
if [ "${SLURM_ARRAY_TASK_ID:-0}" -ge "${#CONFIGS[@]}" ]; then
  echo "task ${SLURM_ARRAY_TASK_ID} is past the end of the ${#CONFIGS[@]}-run list; nothing to do"
  exit 0
fi
read -r ARM TARGET TRAIN_SUBSET SEED <<< "${CONFIGS[${SLURM_ARRAY_TASK_ID:-0}]}"

mkdir -p "$OUTPUT_DIR"

source "$CONDA_ROOT/bin/activate"
conda activate "$CONDA_ENV"

echo "job=${SLURM_JOB_ID:-local} task=${SLURM_ARRAY_TASK_ID:-0} node=${SLURMD_NODENAME:-?}"
echo "arm=$ARM target=$TARGET subset=$TRAIN_SUBSET seed=$SEED (${#CONFIGS[@]} runs total)"

cd "$REPO"
export PYTHONUNBUFFERED=1
# Called directly rather than through srun: the batch task already holds its own
# allocation, and an inner step has to request the GPU again, which fails outright when
# the per-user GPU quota is busy elsewhere.
python -m benchmark.train_pgen_arm \
  --arm "$ARM" \
  --target "$TARGET" \
  --train-subset "$TRAIN_SUBSET" \
  --seed "$SEED" \
  --dataset-dir "$DATASET_DIR" \
  --representations-dir "$REPRESENTATIONS_DIR" \
  --pretrained-checkpoint "$PRETRAINED_CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --patience "$PATIENCE" \
  --threads "${SLURM_CPUS_PER_TASK:-4}"
