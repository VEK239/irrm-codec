#!/bin/bash
#SBATCH --job-name=irrm-ext-preflight
#SBATCH --partition=medium
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/home/evlasova/irrm-codec/experiments/trb/external-model-benchmark/logs/preflight-%j.out
#SBATCH --error=/home/evlasova/irrm-codec/experiments/trb/external-model-benchmark/logs/preflight-%j.err

set -euo pipefail

REPO=/home/evlasova/irrm-codec
PYTHON=/home/evlasova/.conda/envs/irrm-codec/bin/python
DATASET="$REPO/data/benchmark/trb"
RTP="$REPO/experiments/trb/latent-sufficiency/evaluation-1426737/embeddings_rtp.npy"
OUT="$REPO/experiments/trb/external-model-benchmark/preflight-${SLURM_JOB_ID}"

mkdir -p "$REPO/experiments/trb/external-model-benchmark/logs" "$OUT"
cd "$REPO"

"$PYTHON" benchmark/preflight_external_model_benchmark.py \
  --dataset-dir "$DATASET" \
  --rtp-embeddings "$RTP" \
  --output "$OUT/PREFLIGHT.json"

sha256sum \
  benchmark/preflight_external_model_benchmark.py \
  slurm/preflight_external_model_benchmark.sbatch \
  "$OUT/PREFLIGHT.json" > "$OUT/HASHES.sha256"

echo "status=accepted"
echo "output=$OUT"
