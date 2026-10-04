#!/bin/bash
#SBATCH --job-name=irrm-ext-cohort
#SBATCH --partition=medium
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=cohort-%j.out
#SBATCH --error=cohort-%j.err

set -euo pipefail
REPO="${RTP_CODEC_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON="${RTP_CODEC_PYTHON:-python}"
RUNTIME="$REPO/experiments/trb/external-model-benchmark/runtime-v2"
SOURCE="$REPO/data/benchmark/trb"
RTP="$REPO/experiments/trb/latent-sufficiency/evaluation-1426737/embeddings_rtp.npy"
OUT="$REPO/experiments/trb/external-model-benchmark/cohort"

export PYTHONPATH="$RUNTIME/site-packages:$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
mkdir -p "$REPO/experiments/trb/external-model-benchmark/logs" "$OUT"
cd "$REPO"

"$PYTHON" -m rtp_codec.benchmarks.datasets.prepare_external_model_cohort \
  --source-dataset-dir "$SOURCE" \
  --rtp-embeddings "$RTP" \
  --output-dir "$OUT" \
  --seed 42

"$PYTHON" -m rtp_codec.benchmarks.representations.cache_external_model_embeddings \
  --dataset-dir "$OUT" \
  --output-dir "$OUT/representations" \
  --encoders onehot \
  --rtp-embeddings "$OUT/representations/rtp.npy" \
  --device cpu \
  --threads "${SLURM_CPUS_PER_TASK:-16}"

"$PYTHON" -m pytest src/tests/test_external_model_benchmark.py -q
sha256sum \
  src/rtp_codec/benchmarks/datasets/prepare_external_model_cohort.py \
  src/rtp_codec/benchmarks/representations/cache_external_model_embeddings.py \
  "$OUT/COHORT.json" \
  "$OUT/representations/representations.json" > "$OUT/PREPARATION_HASHES.sha256"
echo "status=accepted"
echo "cohort=$OUT"
