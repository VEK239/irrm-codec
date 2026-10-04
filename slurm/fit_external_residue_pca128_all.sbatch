#!/bin/bash -l
#SBATCH --job-name=irrm-res-pca128
#SBATCH --partition=medium
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-2%2
#SBATCH --output=/home/evlasova/irrm-codec/experiments/trb/external-model-benchmark/residue-pca128-v1/logs/pca-%A-%a.out
#SBATCH --error=/home/evlasova/irrm-codec/experiments/trb/external-model-benchmark/residue-pca128-v1/logs/pca-%A-%a.err

set -euo pipefail
REPO="${IRRM_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON=/home/evlasova/.conda/envs/irrm-codec/bin/python
BASE="$REPO/experiments/trb/external-model-benchmark"
ROOT="$BASE/residue-pca128-v1"
REPRESENTATIONS=(esm2_8m tcr_bert sceptr)
REPRESENTATION="${REPRESENTATIONS[$SLURM_ARRAY_TASK_ID]}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
mkdir -p "$ROOT/logs" "$ROOT/bottleneck"
cd "$REPO"
"$PYTHON" benchmark/fit_external_residue_pca256.py \
  --dataset-dir "$BASE/cohort" --residue-dir "$BASE/residue-v1/representations" \
  --output-dir "$ROOT/bottleneck" --representation "$REPRESENTATION" \
  --components 128 --oversample 32 --power-iterations 4 --batch-size 2048 --seed 42
echo "status=accepted representation=$REPRESENTATION"
