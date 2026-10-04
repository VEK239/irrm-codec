#!/bin/bash
#SBATCH --job-name=irrm-ext-preflight
#SBATCH --partition=medium
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=preflight-%j.out
#SBATCH --error=preflight-%j.err

set -euo pipefail

REPO="${RTP_CODEC_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON="${RTP_CODEC_PYTHON:-python}"
DATASET="$REPO/data/benchmark/trb"
RTP="$REPO/experiments/trb/latent-sufficiency/evaluation-1426737/embeddings_rtp.npy"
OUT="$REPO/experiments/trb/external-model-benchmark/preflight-${SLURM_JOB_ID}"

mkdir -p "$REPO/experiments/trb/external-model-benchmark/logs" "$OUT"
cd "$REPO"

"$PYTHON" -m rtp_codec.benchmarks.representations.preflight_external_model_benchmark \
  --dataset-dir "$DATASET" \
  --rtp-embeddings "$RTP" \
  --output "$OUT/PREFLIGHT.json"

sha256sum \
  src/rtp_codec/benchmarks/representations/preflight_external_model_benchmark.py \
  scripts/slurm/preflight/preflight_external_model_benchmark.sbatch \
  "$OUT/PREFLIGHT.json" > "$OUT/HASHES.sha256"

echo "status=accepted"
echo "output=$OUT"
