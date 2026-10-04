#!/bin/bash
#SBATCH --job-name=irrm-author-global-pca
#SBATCH --partition=medium
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/home/evlasova/irrm-codec/experiments/trb/external-global-author-v1/logs/pca-%j.out
#SBATCH --error=/home/evlasova/irrm-codec/experiments/trb/external-global-author-v1/logs/pca-%j.err

set -euo pipefail
REPO=/home/evlasova/irrm-codec
PYTHON=/home/evlasova/.conda/envs/irrm-codec/bin/python
OLD="$REPO/experiments/trb/external-model-benchmark"
BASE="$REPO/experiments/trb/external-global-author-v1"
OUT="$BASE/bottleneck"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
mkdir -p "$BASE/logs" "$OUT"
cd "$REPO"

"$PYTHON" benchmark/fit_external_model_bottleneck.py --dataset-dir "$OLD/cohort" --representations-dir "$OLD/cohort/representations" --output-dir "$OUT" --dims 64 --representations esm2_35m sceptr_cdr3 --batch-size 4096 --seed 42
sha256sum "$OUT"/*_64.npy "$OUT/bottleneck_summary.json" > "$OUT/HASHES.sha256"
echo "status=accepted output=$OUT"
