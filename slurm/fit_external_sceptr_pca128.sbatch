#!/bin/bash -l
#SBATCH --job-name=irrm-sceptr-pca128
#SBATCH --partition=medium
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/evlasova/irrm-codec/experiments/trb/external-model-benchmark/residue-pca128-sceptr-v1/logs/pca-%j.out
#SBATCH --error=/home/evlasova/irrm-codec/experiments/trb/external-model-benchmark/residue-pca128-sceptr-v1/logs/pca-%j.err

set -euo pipefail
REPO="${IRRM_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON=/home/evlasova/.conda/envs/irrm-codec/bin/python
BASE="$REPO/experiments/trb/external-model-benchmark"
ROOT="$BASE/residue-pca128-sceptr-v1"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
mkdir -p "$ROOT/logs" "$ROOT/bottleneck"
cd "$REPO"
"$PYTHON" benchmark/fit_external_residue_pca256.py \
  --dataset-dir "$BASE/cohort" \
  --residue-dir "$BASE/residue-v1/representations" \
  --output-dir "$ROOT/bottleneck" --representation sceptr \
  --components 128 --oversample 32 --power-iterations 4 \
  --batch-size 2048 --seed 42
echo status=accepted
