#!/bin/bash
#SBATCH --job-name=irrm-ext-tfidf
#SBATCH --partition=medium
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/home/evlasova/irrm-codec/experiments/trb/external-model-benchmark/logs/tfidf-%j.out
#SBATCH --error=/home/evlasova/irrm-codec/experiments/trb/external-model-benchmark/logs/tfidf-%j.err

set -euo pipefail
REPO="${IRRM_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON=/home/evlasova/.conda/envs/irrm-codec/bin/python
ROOT="$REPO/experiments/trb/external-model-benchmark"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
mkdir -p "$ROOT/logs" "$ROOT/pgen"
cd "$REPO"

"$PYTHON" benchmark/train_external_model_pgen.py \
  --dataset-dir "$ROOT/cohort" \
  --representations-dir "$ROOT/cohort/representations" \
  --output-dir "$ROOT/pgen" \
  --arm tfidf_ridge \
  --target log10_pgen_1mm \
  --train-subset all \
  --seed 42 \
  --threads "${SLURM_CPUS_PER_TASK:-16}"
echo "status=accepted arm=tfidf_ridge"
