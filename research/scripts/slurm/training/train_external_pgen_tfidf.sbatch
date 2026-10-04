#!/bin/bash
#SBATCH --job-name=irrm-ext-tfidf
#SBATCH --partition=medium
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=tfidf-%j.out
#SBATCH --error=tfidf-%j.err

set -euo pipefail
REPO="${RTP_CODEC_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON="${RTP_CODEC_PYTHON:-python}"
ROOT="$REPO/experiments/trb/external-model-benchmark"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
mkdir -p "$ROOT/logs" "$ROOT/pgen"
cd "$REPO"

"$PYTHON" -m rtp_codec.benchmarks.pgen.train_external_model_pgen \
  --dataset-dir "$ROOT/cohort" \
  --representations-dir "$ROOT/cohort/representations" \
  --output-dir "$ROOT/pgen" \
  --arm tfidf_ridge \
  --target log10_pgen_1mm \
  --train-subset all \
  --seed 42 \
  --threads "${SLURM_CPUS_PER_TASK:-16}"
echo "status=accepted arm=tfidf_ridge"
