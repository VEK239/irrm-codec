#!/bin/bash -l
#SBATCH --job-name=irrm-sceptr-pca128-heads
#SBATCH --partition=medium
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-5%2
#SBATCH --output=/home/evlasova/irrm-codec/experiments/trb/external-model-benchmark/residue-pca128-sceptr-v1/logs/head-%A-%a.out
#SBATCH --error=/home/evlasova/irrm-codec/experiments/trb/external-model-benchmark/residue-pca128-sceptr-v1/logs/head-%A-%a.err

set -euo pipefail
REPO="${IRRM_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON=/home/evlasova/.conda/envs/irrm-codec/bin/python
BASE="$REPO/experiments/trb/external-model-benchmark"
ROOT="$BASE/residue-pca128-sceptr-v1"
SEEDS=(42 43 44)
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
mkdir -p "$ROOT/logs" "$ROOT/reconstruction" "$ROOT/pgen"
cd "$REPO"

if (( SLURM_ARRAY_TASK_ID < 3 )); then
  SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
  "$PYTHON" benchmark/train_external_model_decoder.py \
    --dataset-dir "$BASE/cohort" --bottleneck-dir "$ROOT/bottleneck" \
    --output-dir "$ROOT/reconstruction" --representation sceptr_residue \
    --bottleneck-dim 128 --train-subset all --seed "$SEED" --epochs 60 \
    --batch-size 256 --lr 0.001 --weight-decay 0.0001 --patience 8 \
    --threads "${SLURM_CPUS_PER_TASK:-4}"
  echo "status=accepted task=reconstruction seed=$SEED"
else
  SEED="${SEEDS[$((SLURM_ARRAY_TASK_ID - 3))]}"
  "$PYTHON" benchmark/train_external_model_pgen.py \
    --dataset-dir "$BASE/cohort" --representations-dir "$ROOT/bottleneck" \
    --output-dir "$ROOT/pgen" --arm sceptr_residue_pca128_mlp \
    --target log10_pgen_1mm --train-subset all --seed "$SEED" --epochs 60 \
    --batch-size 256 --lr 0.001 --weight-decay 0.0001 --patience 12 \
    --threads "${SLURM_CPUS_PER_TASK:-4}"
  echo "status=accepted task=pgen seed=$SEED"
fi
