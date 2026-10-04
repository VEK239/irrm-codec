#!/bin/bash
#SBATCH --job-name=irrm-ext-pca
#SBATCH --partition=medium
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --output=pca-%j.out
#SBATCH --error=pca-%j.err

set -euo pipefail
REPO="${RTP_CODEC_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON="${RTP_CODEC_PYTHON:-python}"
ROOT="$REPO/experiments/trb/external-model-benchmark"
COHORT="$ROOT/cohort"
OUT="$ROOT/bottleneck"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
mkdir -p "$ROOT/logs" "$OUT"
cd "$REPO"

"$PYTHON" -m rtp_codec.benchmarks.representations.fit_external_model_bottleneck \
  --dataset-dir "$COHORT" \
  --representations-dir "$COHORT/representations" \
  --output-dir "$OUT" \
  --dims 64 \
  --representations onehot tcremp esm2_8m tcr_bert sceptr rtp \
  --batch-size 4096 \
  --seed 42

sha256sum "$OUT"/*_64.npy "$OUT/bottleneck_summary.json" > "$OUT/HASHES.sha256"
echo "status=accepted"
