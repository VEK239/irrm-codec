#!/bin/bash
#SBATCH --job-name=irrm-ext-pca
#SBATCH --partition=medium
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --output=/home/evlasova/irrm-codec/experiments/trb/external-model-benchmark/logs/pca-%j.out
#SBATCH --error=/home/evlasova/irrm-codec/experiments/trb/external-model-benchmark/logs/pca-%j.err

set -euo pipefail
REPO=/home/evlasova/irrm-codec
PYTHON=/home/evlasova/.conda/envs/irrm-codec/bin/python
ROOT="$REPO/experiments/trb/external-model-benchmark"
COHORT="$ROOT/cohort"
OUT="$ROOT/bottleneck"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
mkdir -p "$ROOT/logs" "$OUT"
cd "$REPO"

"$PYTHON" benchmark/fit_external_model_bottleneck.py \
  --dataset-dir "$COHORT" \
  --representations-dir "$COHORT/representations" \
  --output-dir "$OUT" \
  --dims 64 \
  --representations onehot tcremp esm2_8m tcr_bert sceptr rtp \
  --batch-size 4096 \
  --seed 42

sha256sum "$OUT"/*_64.npy "$OUT/bottleneck_summary.json" > "$OUT/HASHES.sha256"
echo "status=accepted"
