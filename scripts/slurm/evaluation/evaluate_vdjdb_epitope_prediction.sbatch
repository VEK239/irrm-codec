#!/bin/bash -l
#SBATCH --job-name=vdjdb-epitope-prediction
#SBATCH --partition=medium
#SBATCH --constraint=hpc
#SBATCH --cpus-per-task=20
#SBATCH --mem=160G
#SBATCH --time=16:00:00
#SBATCH --output=run-%j.out
#SBATCH --error=run-%j.err

set -euo pipefail
REPO="${RTP_CODEC_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON="${RTP_CODEC_PYTHON:-python}"
BASE="$REPO/experiments/trb/epitope-prediction-v1"
COHORT="$REPO/experiments/trb/vdjdb-redcea-clonotypes/preflight-1426719/cohort/cohort.tsv"
UNIFORM="$REPO/experiments/trb/vdjdb-uniform100-v1/evaluation-1427813"
NATIVE="$REPO/experiments/trb/vdjdb-redcea-clonotypes/evaluation-1426728"
EXTERNAL="$REPO/experiments/trb/external-global-author-v1/vdjdb/representations"
OUT="${EPITOPE_PREDICTION_OUT:-$BASE/evaluation-$SLURM_JOB_ID}"

export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
mkdir -p "$BASE/logs"
cd "$REPO"

"$PYTHON" -m rtp_codec.benchmarks.downstream.evaluate_vdjdb_epitope_prediction \
  --cohort "$COHORT" \
  --uniform-evaluation "$UNIFORM" \
  --native-evaluation "$NATIVE" \
  --external-representations "$EXTERNAL" \
  --output-dir "$OUT" \
  --seeds 42 43 44 \
  --folds 5 \
  --c-grid 0.1 1.0 \
  --max-iter 400 \
  --jobs 16

sha256sum "$OUT"/* > "$OUT/HASHES.sha256"
echo "status=complete output=$OUT"
